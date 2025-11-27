"""
PokeProtocol implementation with:
- UDP protocol with Host / Joiner / Spectator roles
- Battle state machine and damage calculation using pokemon.csv
- Browser GUI (Flask + SocketIO) for Host/Joiner/Spectator
- Spectator sticker support (base64, 320x320, <10MB)

This version removes the complex reliability/retransmission layer to avoid
resend loops and focuses on a clean, working protocol flow while keeping
message types and text format compatible with the RFC-style spec.

Run examples (from folder containing this file and pokemon.csv):

Host:
    python pokeprotocol.py --role host --udp-port 50010 --http-port 8000 --name HOST

Joiner (same machine):
    python pokeprotocol.py --role joiner --host-ip 127.0.0.1 --host-port 50010 --udp-port 50011 --http-port 8001 --name JOINER

Spectator with GUI:
    python pokeprotocol.py --role spectator --host-ip 127.0.0.1 --host-port 50010 --udp-port 50012 --http-port 8002 --name VIEWER
"""

import base64
import csv
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple, List, Callable

# ===================== Pokémon model & CSV loader =====================

POKEMON_CSV_PATH = "pokemon.csv"


@dataclass
class Pokemon:
    name: str
    type1: str
    type2: Optional[str]
    hp: int
    attack: int
    sp_attack: int
    defense: int
    sp_defense: int
    against: Dict[str, float] = field(default_factory=dict)


def load_pokemon_csv(path: str = POKEMON_CSV_PATH) -> Dict[str, Pokemon]:
    """
    Load pokemon.csv and return mapping: name.lower() -> Pokemon
    Expects columns:
      name, type1, type2, hp, attack, sp_attack, defense, sp_defense,
      plus columns against_<type> (e.g., against_fire, against_water, ...)
    """
    pokedex: Dict[str, Pokemon] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"pokemon.csv not found at: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row["name"]
                type1 = row.get("type1", "").lower()
                type2_raw = row.get("type2", "")
                type2 = type2_raw.lower() if type2_raw and type2_raw != "None" else None

                hp = int(float(row["hp"]))
                attack = int(float(row["attack"]))
                sp_attack = int(float(row["sp_attack"]))
                defense = int(float(row["defense"]))
                sp_defense = int(float(row["sp_defense"]))

                against: Dict[str, float] = {}
                for key, val in row.items():
                    if key.startswith("against_"):
                        t = key[len("against_") :].lower()
                        try:
                            against[t] = float(val)
                        except (TypeError, ValueError):
                            against[t] = 1.0

                pokedex[name.lower()] = Pokemon(
                    name=name,
                    type1=type1,
                    type2=type2,
                    hp=hp,
                    attack=attack,
                    sp_attack=sp_attack,
                    defense=defense,
                    sp_defense=sp_defense,
                    against=against,
                )
            except Exception:
                # Skip malformed rows
                continue
    return pokedex


# ===================== Protocol encoding helpers =====================


def encode_message(fields: Dict[str, str]) -> bytes:
    """
    Encode key/value fields as lines "key: value" separated by newlines.
    """
    lines = []
    for k, v in fields.items():
        lines.append(f"{k}: {v}")
    return ("\n".join(lines)).encode("utf-8")


def decode_message(data: bytes) -> Dict[str, str]:
    """
    Decode UDP payload to dict of key/value pairs.
    """
    text = data.decode("utf-8", errors="replace").strip()
    result: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        result[key.strip()] = value.strip()
    return result


class GameState(Enum):
    SETUP = auto()
    WAITING_FOR_MOVE = auto()
    PROCESSING_TURN = auto()
    GAME_OVER = auto()


@dataclass
class StatBoosts:
    special_attack_uses: int = 5
    special_defense_uses: int = 5


@dataclass
class BattleSide:
    pokemon: Pokemon
    current_hp: int
    stat_boosts: StatBoosts = field(default_factory=StatBoosts)


# ===================== Damage calculation =====================


def get_type_effectiveness(attacker: Pokemon, defender_type: Optional[str]) -> float:
    if defender_type is None:
        return 1.0
    return attacker.against.get(defender_type.lower(), 1.0)


def calculate_damage(
    attacker: BattleSide,
    defender: BattleSide,
    use_special_attack: bool = False,
    use_special_defense: bool = False,
) -> Tuple[int, int]:
    """
    Damage formula:
        (BasePower x AttackerStat x Type1Effectiveness x Type2Effectiveness) / DefenderStat

    BasePower = 1
    AttackerStat = attack or sp_attack (if special attack use is consumed)
    DefenderStat = defense or sp_defense (if special defense use is consumed)
    """
    atk_poke = attacker.pokemon
    def_poke = defender.pokemon

    attacker_stat = atk_poke.sp_attack if use_special_attack else atk_poke.attack
    defender_stat = def_poke.sp_defense if use_special_defense else def_poke.defense
    if defender_stat <= 0:
        defender_stat = 1

    t1 = get_type_effectiveness(atk_poke, def_poke.type1)
    t2 = get_type_effectiveness(atk_poke, def_poke.type2)

    raw_damage = (1.0 * attacker_stat * t1 * t2) / defender_stat
    damage = max(1, int(round(raw_damage)))
    new_hp = max(0, defender.current_hp - damage)
    return damage, new_hp


# ===================== Core PokePeer =====================


class Role(Enum):
    HOST = "host"
    JOINER = "joiner"
    SPECTATOR = "spectator"


class PokePeer:
    """
    Core PokeProtocol node.
    Handles:
      - UDP send/recv
      - Handshake (Host/Joiner/Spectator)
      - Battle state & damage resolution
      - Chat messages (text + stickers)
      - Simple sequence numbers (no retransmission loops)
    """

    def __init__(
        self,
        role: Role,
        udp_port: int,
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        seed: Optional[int] = None,
        on_debug: Optional[Callable[[str], None]] = None,
        on_chat: Optional[Callable[[Dict[str, str]], None]] = None,
    ):
        self.role = role
        self.udp_port = udp_port
        self.remote_addr: Optional[Tuple[str, int]] = (
            (remote_host, remote_port) if remote_host and remote_port else None
        )
        self.seed = seed
        self._debug_cb = on_debug or print
        self._chat_cb = on_chat or (lambda x: None)

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", udp_port))
        self.sock.settimeout(0.1)

        # Spectators (host only)
        self.spectators: List[Tuple[str, int]] = []

        # Simple sequence number for messages that use it (no retransmit logic)
        self._next_seq = 1

        # Battle state
        self.state: GameState = GameState.SETUP
        self.my_side: Optional[BattleSide] = None
        self.opponent_side: Optional[BattleSide] = None
        self.communication_mode: str = "P2P"
        self.is_my_turn: bool = False
        self.last_move_name: str = ""
        self.battle_setup_sent: bool = False

        # Threading
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None

    # ---------- debug ----------

    def debug(self, msg: str) -> None:
        self._debug_cb(msg)

    def chat(self, fields: Dict[str, str]) -> None:
        """Emit chat message to GUI"""
        self._chat_cb(fields)

    # ---------- sequence helper ----------

    def next_sequence_number(self) -> int:
        s = self._next_seq
        self._next_seq += 1
        return s

    # ---------- low-level send ----------

    def _send_raw(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            self.sock.sendto(data, addr)
        except OSError as e:
            self.debug(f"[send_raw error] {e}")

    def _send_with_seq(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        seq = self.next_sequence_number()
        fields = dict(fields)
        fields["sequence_number"] = str(seq)
        self._send_raw(encode_message(fields), addr)

    # ---------- broadcast from host ----------

    def host_broadcast(self, fields: Dict[str, str]) -> None:
        """
        Host sends a message to Joiner (remote_addr) and all spectators.
        No retransmission; just best-effort UDP.
        """
        if self.role != Role.HOST:
            return
        targets: List[Tuple[str, int]] = []
        if self.remote_addr:
            targets.append(self.remote_addr)
        targets.extend(self.spectators)
        for addr in targets:
            self._send_with_seq(dict(fields), addr)

    # ---------- run loop ----------

    def start(self) -> None:
        self.running = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def stop(self) -> None:
        self.running = False
        try:
            if self.recv_thread and self.recv_thread.is_alive():
                self.recv_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def _recv_loop(self) -> None:
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            fields = decode_message(data)
            self._handle_incoming(fields, addr)

    # ---------- message handling ----------

    def _handle_incoming(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        msg_type = fields.get("message_type", "")

        # ---- Handshakes ----
        if msg_type == "HANDSHAKE_REQUEST" and self.role == Role.HOST:
            # Joiner is main opponent
            self.remote_addr = addr
            self.debug(f"[HOST] Joiner detected at {addr}")
            resp = {
                "message_type": "HANDSHAKE_RESPONSE",
                "seed": str(self.seed or 12345),
            }
            self._send_raw(encode_message(resp), addr)
            return

        if msg_type == "SPECTATOR_REQUEST" and self.role == Role.HOST:
            if addr not in self.spectators:
                self.spectators.append(addr)
                self.debug(f"[HOST] New spectator at {addr}")
            resp = {
                "message_type": "HANDSHAKE_RESPONSE",
                "seed": str(self.seed or 12345),
            }
            self._send_raw(encode_message(resp), addr)
            return

        if msg_type == "HANDSHAKE_RESPONSE":
            seed_str = fields.get("seed")
            try:
                self.seed = int(seed_str)
            except (TypeError, ValueError):
                self.seed = None
            self.debug(f"[{self.role.value.upper()}] Handshake response, seed={self.seed}")
            return

        # ---- Battle and chat ----
        if msg_type == "BATTLE_SETUP":
            self._handle_battle_setup(fields, addr)
        elif msg_type == "ATTACK_ANNOUNCE":
            self._handle_attack_announce(fields, addr)
        elif msg_type == "CALCULATION_REPORT":
            self._handle_calculation_report(fields, addr)
        elif msg_type == "GAME_OVER":
            self._handle_game_over(fields, addr)
        elif msg_type == "CHAT_MESSAGE":
            self._handle_chat_message(fields, addr)
        else:
            self.debug(f"[RECV] Unknown message_type from {addr}: {msg_type}")

    # ---------- handlers ----------

    def _handle_battle_setup(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        self.debug(f"[RECV] BATTLE_SETUP from {addr}")
        self.communication_mode = fields.get("communication_mode", "P2P")
        pokemon_json = fields.get("pokemon", "{}")
        boosts_json = fields.get("stat_boosts", "{}")
        try:
            p_dict = json.loads(pokemon_json)
            p = Pokemon(
                name=p_dict["name"],
                type1=p_dict["type1"],
                type2=p_dict.get("type2"),
                hp=int(p_dict["hp"]),
                attack=int(p_dict["attack"]),
                sp_attack=int(p_dict["sp_attack"]),
                defense=int(p_dict["defense"]),
                sp_defense=int(p_dict["sp_defense"]),
                against={k: float(v) for k, v in p_dict.get("against", {}).items()},
            )
            boosts_parsed = json.loads(boosts_json or "{}")
            boosts = StatBoosts(
                special_attack_uses=int(boosts_parsed.get("special_attack_uses", 5)),
                special_defense_uses=int(boosts_parsed.get("special_defense_uses", 5)),
            )
            # For spectators: use my_side for first pokemon, opponent_side for second
            # For host/joiner: opponent_side is the other player's pokemon
            if self.role == Role.SPECTATOR:
                if self.my_side is None:
                    self.my_side = BattleSide(pokemon=p, current_hp=p.hp, stat_boosts=boosts)
                    self.debug(f"[STATE] Player 1 Pokémon: {p.name} ({p.type1}/{p.type2}), HP={p.hp}")
                elif self.opponent_side is None:
                    self.opponent_side = BattleSide(pokemon=p, current_hp=p.hp, stat_boosts=boosts)
                    self.debug(f"[STATE] Player 2 Pokémon: {p.name} ({p.type1}/{p.type2}), HP={p.hp}")
            else:
                self.opponent_side = BattleSide(pokemon=p, current_hp=p.hp, stat_boosts=boosts)
                self.debug(
                    f"[STATE] Opponent Pokémon: {p.name} ({p.type1}/{p.type2}), "
                    f"HP={p.hp}, atk={p.attack}, sp_atk={p.sp_attack}, "
                    f"def={p.defense}, sp_def={p.sp_defense}"
                )
            # Transition to battle ready state
            if self.state == GameState.SETUP:
                # For spectators: wait for both pokemon
                if self.role == Role.SPECTATOR:
                    if self.my_side is not None and self.opponent_side is not None:
                        self.state = GameState.WAITING_FOR_MOVE
                        self.debug("[STATE] Battle ready (spectator view)")
                # For host/joiner: need own pokemon and opponent's
                elif self.my_side is not None:
                    self.state = GameState.WAITING_FOR_MOVE
                    if self.role == Role.HOST:
                        self.is_my_turn = True
                    else:
                        self.is_my_turn = False
                    self.debug(f"[STATE] Battle ready. My turn? {self.is_my_turn}")
        except Exception as e:
            self.debug(f"[ERROR] Failed to parse BATTLE_SETUP: {e}")

        # Forward BATTLE_SETUP to spectators so they can see the battle state
        if self.role == Role.HOST:
            self._broadcast_to_spectators(fields)

    def _handle_attack_announce(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        move_name = fields.get("move_name", "Unknown Move")
        self.last_move_name = move_name
        self.debug(f"[RECV] ATTACK_ANNOUNCE from {addr}: move={move_name}")
        # Host forwards announce to spectators
        if self.role == Role.HOST:
            self._broadcast_to_spectators(fields)

    def _handle_calculation_report(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        self.debug(f"[RECV] CALCULATION_REPORT from {addr}: {fields}")
        if self.role == Role.HOST:
            self._broadcast_to_spectators(fields)

        try:
            defender_hp_remaining = int(fields.get("defender_hp_remaining", "0"))
            attacker_name = fields.get("attacker", "")
            if self.my_side and attacker_name == self.my_side.pokemon.name:
                # We attacked; opponent is defender
                if self.opponent_side:
                    self.opponent_side.current_hp = defender_hp_remaining
            else:
                # Opponent attacked; we are defender
                if self.my_side:
                    self.my_side.current_hp = defender_hp_remaining
        except ValueError:
            pass

        # Update state/turn
        if self.my_side and self.my_side.current_hp <= 0:
            self.state = GameState.GAME_OVER
        elif self.opponent_side and self.opponent_side.current_hp <= 0:
            self.state = GameState.GAME_OVER
        else:
            self.state = GameState.WAITING_FOR_MOVE
            self.is_my_turn = not self.is_my_turn

    def _handle_game_over(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        winner = fields.get("winner", "")
        loser = fields.get("loser", "")
        self.debug(f"[RECV] GAME_OVER from {addr}: winner={winner}, loser={loser}")
        if self.role == Role.HOST:
            self._broadcast_to_spectators(fields)
        self.state = GameState.GAME_OVER

    def _handle_chat_message(self, fields: Dict[str, str], addr: Tuple[str, int]) -> None:
        sender_name = fields.get("sender_name", "Unknown")
        content_type = fields.get("content_type", "TEXT")
        if content_type == "TEXT":
            text = fields.get("message_text", "")
            self.debug(f"[CHAT] ({sender_name}) {text}")
        elif content_type == "STICKER":
            sticker_data = fields.get("sticker_data", "")
            self.debug(f"[CHAT] ({sender_name}) [STICKER len={len(sticker_data)}]")
        else:
            self.debug(f"[CHAT] ({sender_name}) [UNKNOWN CONTENT_TYPE={content_type}]")

        # Emit chat message to GUI
        self.chat(fields)

        if self.role == Role.HOST:
            # If message is from a spectator, broadcast to everyone (joiner + spectators)
            # If message is from joiner, only broadcast to spectators (joiner already showed it)
            if addr in self.spectators:
                # From spectator - broadcast to joiner and all spectators
                self.host_broadcast(fields)
            else:
                # From joiner - only broadcast to spectators
                self._broadcast_to_spectators(fields)

    def _broadcast_to_spectators(self, fields: Dict[str, str]) -> None:
        if self.role != Role.HOST or not self.spectators:
            return
        for saddr in self.spectators:
            self._send_with_seq(dict(fields), saddr)

    # ---------- high-level API (used by GUI / external code) ----------

    def send_handshake_request(self) -> None:
        """
        Joiner/Spectator -> Host: initial handshake.
        """
        if not self.remote_addr:
            self.debug("[WARN] remote_addr not set; cannot send handshake.")
            return
        if self.role == Role.JOINER:
            fields = {"message_type": "HANDSHAKE_REQUEST"}
        elif self.role == Role.SPECTATOR:
            fields = {"message_type": "SPECTATOR_REQUEST"}
        else:
            return
        self._send_raw(encode_message(fields), self.remote_addr)

    def send_battle_setup(self) -> None:
        if self.battle_setup_sent:
            self.debug("[WARN] BATTLE_SETUP already sent. Cannot send again.")
            return
        if not self.remote_addr:
            self.debug("[WARN] Tried to send BATTLE_SETUP but remote_addr is not set yet.")
            return
        if not self.my_side:
            self.debug("[WARN] Tried to send BATTLE_SETUP but my_side is not set.")
            return

        p = self.my_side.pokemon
        p_dict = {
            "name": p.name,
            "type1": p.type1,
            "type2": p.type2,
            "hp": p.hp,
            "attack": p.attack,
            "sp_attack": p.sp_attack,
            "defense": p.defense,
            "sp_defense": p.sp_defense,
            "against": p.against,
        }
        boosts = {
            "special_attack_uses": self.my_side.stat_boosts.special_attack_uses,
            "special_defense_uses": self.my_side.stat_boosts.special_defense_uses,
        }

        fields = {
            "message_type": "BATTLE_SETUP",
            "communication_mode": self.communication_mode,
            "pokemon_name": p.name,
            "stat_boosts": json.dumps(boosts),
            "pokemon": json.dumps(p_dict),
        }
        # Send to opponent and spectators (if host)
        if self.role == Role.HOST:
            self.host_broadcast(fields)
        else:
            self._send_with_seq(fields, self.remote_addr)

        # Mark that battle setup has been sent
        self.battle_setup_sent = True

    def announce_attack(
        self,
        move_name: str,
        use_special_attack: bool = False,
        use_special_defense: bool = False,
    ) -> None:
        """
        One-step attack:
          - Send ATTACK_ANNOUNCE (for UI/log)
          - Locally compute damage and update HP
          - Send CALCULATION_REPORT
          - If defender HP <= 0, send GAME_OVER
        """
        if self.state != GameState.WAITING_FOR_MOVE or not self.is_my_turn:
            self.debug("[INFO] Cannot attack: not your turn or wrong state.")
            return
        if not self.remote_addr:
            self.debug("[WARN] Cannot attack: remote_addr not set.")
            return
        if not self.my_side or not self.opponent_side:
            self.debug("[ERROR] My or opponent side not set.")
            return

        self.last_move_name = move_name or "Attack"

        if use_special_attack and self.my_side.stat_boosts.special_attack_uses <= 0:
            self.debug("[WARN] No special attack uses left.")
            use_special_attack = False
        if use_special_defense and self.my_side.stat_boosts.special_defense_uses <= 0:
            self.debug("[WARN] No special defense uses left.")
            use_special_defense = False

        # Consume boosts
        if use_special_attack:
            self.my_side.stat_boosts.special_attack_uses -= 1
        if use_special_defense:
            self.my_side.stat_boosts.special_defense_uses -= 1

        # ATTACK_ANNOUNCE (for log/spectators)
        announce = {
            "message_type": "ATTACK_ANNOUNCE",
            "move_name": self.last_move_name,
        }
        if self.role == Role.HOST:
            self.host_broadcast(announce)
        else:
            self._send_with_seq(announce, self.remote_addr)

        # Local damage calc
        self.state = GameState.PROCESSING_TURN
        damage, new_hp = calculate_damage(
            attacker=self.my_side,
            defender=self.opponent_side,
            use_special_attack=use_special_attack,
            use_special_defense=use_special_defense,
        )
        self.opponent_side.current_hp = new_hp

        status_msg = f"{self.my_side.pokemon.name} used {self.last_move_name}! Damage: {damage}."
        report = {
            "message_type": "CALCULATION_REPORT",
            "attacker": self.my_side.pokemon.name,
            "move_used": self.last_move_name,
            "remaining_health": str(self.my_side.current_hp),
            "damage_dealt": str(damage),
            "defender_hp_remaining": str(new_hp),
            "status_message": status_msg,
        }
        if self.role == Role.HOST:
            self.host_broadcast(report)
        else:
            self._send_with_seq(report, self.remote_addr)

        # Game-over check
        if new_hp <= 0:
            game_over = {
                "message_type": "GAME_OVER",
                "winner": self.my_side.pokemon.name,
                "loser": self.opponent_side.pokemon.name,
            }
            if self.role == Role.HOST:
                self.host_broadcast(game_over)
            else:
                self._send_with_seq(game_over, self.remote_addr)
            self.state = GameState.GAME_OVER
        else:
            self.state = GameState.WAITING_FOR_MOVE
            self.is_my_turn = False

    def send_chat_text(self, sender_name: str, text: str) -> None:
        if not text:
            return
        fields = {
            "message_type": "CHAT_MESSAGE",
            "sender_name": sender_name,
            "content_type": "TEXT",
            "message_text": text,
        }
        # Show the message in sender's own GUI (but not for spectators - they'll receive it back from host)
        if self.role != Role.SPECTATOR:
            self.chat(fields)

        if self.role == Role.HOST:
            self.host_broadcast(fields)
        else:
            if self.remote_addr:
                self._send_with_seq(fields, self.remote_addr)

    def send_chat_sticker(self, sender_name: str, image_bytes: bytes) -> None:
        if len(image_bytes) >= 10 * 1024 * 1024:
            self.debug("[ERROR] Sticker too large (>=10MB)")
            return
        b64 = base64.b64encode(image_bytes).decode("ascii")
        fields = {
            "message_type": "CHAT_MESSAGE",
            "sender_name": sender_name,
            "content_type": "STICKER",
            "sticker_data": b64,
        }
        # Show the sticker in sender's own GUI (but not for spectators - they'll receive it back from host)
        if self.role != Role.SPECTATOR:
            self.chat(fields)

        if self.role == Role.HOST:
            self.host_broadcast(fields)
        else:
            if self.remote_addr:
                self._send_with_seq(fields, self.remote_addr)

    def get_state_summary(self) -> Dict[str, object]:
        return {
            "state": self.state.name,
            "is_my_turn": self.is_my_turn,
            "my_pokemon": self.my_side.pokemon.name if self.my_side else None,
            "my_hp": self.my_side.current_hp if self.my_side else None,
            "my_max_hp": self.my_side.pokemon.hp if self.my_side else None,
            "opp_pokemon": self.opponent_side.pokemon.name if self.opponent_side else None,
            "opp_hp": self.opponent_side.current_hp if self.opponent_side else None,
            "opp_max_hp": self.opponent_side.pokemon.hp if self.opponent_side else None,
            "my_special_attack_uses": self.my_side.stat_boosts.special_attack_uses
            if self.my_side
            else None,
            "my_special_defense_uses": self.my_side.stat_boosts.special_defense_uses
            if self.my_side
            else None,
            "last_move_name": self.last_move_name,
        }


# ===================== Browser GUI (Flask + SocketIO) =====================


def run_gui(peer: PokePeer, pokedex: Dict[str, Pokemon], http_port: int, display_name: str):
    from flask import Flask, jsonify
    from flask_socketio import SocketIO, emit

    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    debug_buffer: List[str] = []

    def gui_debug(msg: str):
        debug_buffer.append(msg)
        if len(debug_buffer) > 300:
            debug_buffer.pop(0)
        print(msg)
        socketio.emit("debug_log", {"line": msg})

    def gui_chat(fields: Dict[str, str]):
        """Emit chat messages to the GUI"""
        socketio.emit("chat", fields)

    peer._debug_cb = gui_debug
    peer._chat_cb = gui_chat

    @app.route("/")
    def index():
        # Entire single-page app as HTML/JS/CSS
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>PokeProtocol GUI ({peer.role.value})</title>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      margin: 0;
      padding: 0;
      background: #f4f5fb;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.2fr 1.5fr 1fr;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      height: 100vh;
      padding: 12px;
      box-sizing: border-box;
    }}
    .card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .card h2 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .pokemon-list {{
      flex: 1;
      overflow-y: auto;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 4px;
    }}
    .pokemon-item {{
      padding: 4px 6px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }}
    .pokemon-item:hover {{
      background: #f0f2ff;
    }}
    .pokemon-item.selected {{
      background: #dce3ff;
      font-weight: 600;
    }}
    .badge {{
      border-radius: 999px;
      padding: 1px 6px;
      font-size: 11px;
      background: #eef;
      margin-left: 4px;
    }}
    .hp-bar {{
      width: 100%;
      height: 14px;
      background: #eee;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
      margin-bottom: 4px;
    }}
    .hp-fill {{
      height: 100%;
      background: linear-gradient(90deg, #4ade80, #22c55e);
      transition: width 0.2s ease;
    }}
    #chat-log {{
      flex: 1;
      border: 1px solid #ddd;
      border-radius: 8px;
      overflow-y: auto;
      padding: 4px;
      font-size: 13px;
      background: #fafbff;
    }}
    .chat-line {{
      margin-bottom: 4px;
      word-wrap: break-word;
    }}
    .chat-sender {{
      font-weight: 600;
    }}
    #debug-log {{
      flex: 1;
      border: 1px solid #ddd;
      border-radius: 8px;
      overflow-y: auto;
      font-size: 11px;
      background: #111827;
      color: #e5e7eb;
      padding: 4px;
      font-family: "JetBrains Mono", monospace;
    }}
    .state-pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: #eff6ff;
      color: #1d4ed8;
    }}
    .turn-pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: #ecfdf3;
      color: #166534;
      margin-left: 6px;
    }}
    button {{
      border-radius: 999px;
      border: none;
      padding: 6px 10px;
      font-size: 13px;
      cursor: pointer;
      background: #4f46e5;
      color: #fff;
    }}
    button.secondary {{
      background: #e5e7eb;
      color: #111827;
    }}
    button.glow {{
      box-shadow: 0 0 6px rgba(79,70,229,0.8);
    }}
    button.toggled {{
      background: #1d4ed8;
      color: #fff;
    }}
    input, textarea {{
      font-size: 13px;
      padding: 4px 6px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      width: 100%;
      box-sizing: border-box;
    }}
    .row {{
      display: flex;
      gap: 6px;
      align-items: center;
      margin-top: 4px;
    }}
    .row > * {{
      flex: 1;
    }}
    .small {{
      font-size: 11px;
      color: #6b7280;
    }}
    .spectator-hide {{
      display: none !important;
    }}
    /* Spectator-specific layout */
    body.spectator-mode .layout {{
      grid-template-columns: 1.5fr 1fr;
      grid-template-rows: auto auto;
    }}
    body.spectator-mode .battle-status-card {{
      grid-column: 1;
      grid-row: 1 / span 2;
    }}
    body.spectator-mode .chat-card {{
      grid-column: 2;
      grid-row: 1;
    }}
    body.spectator-mode .debug-card {{
      grid-column: 2;
      grid-row: 2;
    }}
  </style>
</head>
<body>
  <div class="layout">
    <!-- Left: Pokémon chooser -->
    <div class="card spectator-only-hide" style="grid-row: 1 / span 3;">
      <h2>1. Choose Pokémon</h2>
      <div class="row">
        <input id="search" placeholder="Search by name or type..." />
      </div>
      <div class="pokemon-list" id="pokemon-list"></div>
      <div class="small" style="margin-top:4px;">
        Selected: <span id="selected-pokemon-label">None</span>
      </div>
      <div class="row" style="margin-top:6px;">
        <button id="btn-setup">Send BATTLE_SETUP</button>
      </div>
      <div class="small">
        Role: <b>{peer.role.value}</b> | You are: <b>{display_name}</b>
      </div>
    </div>

    <!-- Middle top: battle status -->
    <div class="card battle-status-card" style="grid-row: 1 / span 1;">
      <h2>2. Battle Status</h2>
      <div id="state-row" class="small">
        <span class="state-pill" id="state-pill">SETUP</span>
        <span class="turn-pill" id="turn-pill">Waiting…</span>
      </div>
      <div style="margin-top:6px;">
        <div class="small">Your Pokémon</div>
        <div id="my-pokemon-name" style="font-weight:600;">None</div>
        <div class="hp-bar">
          <div class="hp-fill" id="my-hp-fill" style="width:0%;"></div>
        </div>
        <div class="small" id="my-hp-label">0 / 0 HP</div>
      </div>
      <div style="margin-top:6px;">
        <div class="small">Opponent Pokémon</div>
        <div id="opp-pokemon-name" style="font-weight:600;">None</div>
        <div class="hp-bar">
          <div class="hp-fill" id="opp-hp-fill" style="width:0%;"></div>
        </div>
        <div class="small" id="opp-hp-label">0 / 0 HP</div>
      </div>
    </div>

    <!-- Middle bottom: attack controls -->
    <div class="card spectator-only-hide" style="grid-row: 2 / span 1;">
      <h2>3. Attack & Boosts</h2>
      <div class="row">
        <input id="move-name" placeholder="Enter move name (just a label)" />
        <button id="btn-attack">ATTACK</button>
      </div>
      <div class="row">
        <button id="btn-special-atk" class="secondary">
          Special ATK (<span id="special-atk-count">5</span>)
        </button>
        <button id="btn-special-def" class="secondary">
          Special DEF (<span id="special-def-count">5</span>)
        </button>
      </div>
      <div class="small" id="attack-help">
        Toggle Special ATK/DEF then hit ATTACK. Button glows when a boost is active.
      </div>
    </div>

    <!-- Right: chat -->
    <div class="card chat-card" style="grid-column: 3; grid-row: 1 / span 2;">
      <h2>4. Chat & Stickers</h2>
      <div id="chat-log"></div>
      <div class="row" style="margin-top:6px;">
        <input id="chat-input" placeholder="Type a message..." />
        <button id="btn-send-chat">Send</button>
      </div>
      <div class="row" style="margin-top:6px;">
        <input type="file" id="sticker-input" accept="image/*" />
        <button class="secondary" id="btn-send-sticker">Send Sticker</button>
      </div>
      <div class="small">
        Stickers must be exactly 320x320 and &lt; 10MB. Validation is done in the browser.
      </div>
    </div>

    <!-- Bottom: debug -->
    <div class="card debug-card" style="grid-column: 2 / span 2; grid-row: 3;">
      <h2>5. Debug (terminal mirror)</h2>
      <div id="debug-log"></div>
    </div>
  </div>

  <script>
    const socket = io();
    const MY_ROLE = '{peer.role.value}';

    // Hide battle controls for spectators and apply spectator layout
    if (MY_ROLE === 'spectator') {{
      document.body.classList.add('spectator-mode');
      document.addEventListener('DOMContentLoaded', () => {{
        document.querySelectorAll('.spectator-only-hide').forEach(el => {{
          el.classList.add('spectator-hide');
        }});
      }});
    }}

    function appendChat(line, isSticker=false, stickerData=null) {{
      const chat = document.getElementById('chat-log');
      const div = document.createElement('div');
      div.className = 'chat-line';
      if (!isSticker) {{
        div.textContent = line;
      }} else {{
        div.innerHTML = line + '<br/><img style="max-width:96px;max-height:96px;border-radius:8px;" src="data:image/png;base64,' +
          stickerData + '"/>';
      }}
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }}

    function appendDebug(line) {{
      const dbg = document.getElementById('debug-log');
      const div = document.createElement('div');
      div.textContent = line;
      dbg.appendChild(div);
      dbg.scrollTop = dbg.scrollHeight;
    }}

    socket.on('connect', () => {{
      appendDebug('[GUI] Connected to backend');
      fetch('/pokemon')
        .then(r => r.json())
        .then(data => {{
          window.__POKEDEX = data.pokemon || [];
          renderPokemonList('');
        }});
    }});

    socket.on('debug_log', (payload) => {{
      if (payload && payload.line) {{
        appendDebug(payload.line);
      }}
    }});

    socket.on('chat', (payload) => {{
      const sender = payload.sender_name || 'Unknown';
      const type = payload.content_type || 'TEXT';
      if (type === 'TEXT') {{
        appendChat(sender + ': ' + (payload.message_text || ''));
      }} else if (type === 'STICKER') {{
        appendChat(sender + ' sent a sticker:', true, payload.sticker_data || '');
      }}
    }});

    socket.on('state', (s) => {{
      if (!s) return;
      document.getElementById('state-pill').textContent = s.state;
      document.getElementById('turn-pill').textContent = s.is_my_turn ? 'Your turn' : 'Opponent turn';

      document.getElementById('my-pokemon-name').textContent = s.my_pokemon || 'None';
      document.getElementById('opp-pokemon-name').textContent = s.opp_pokemon || 'None';

      function setHp(fillId, labelId, hp, maxHp) {{
        const fill = document.getElementById(fillId);
        const label = document.getElementById(labelId);
        const pct = (hp && maxHp) ? Math.max(0, Math.min(100, hp / maxHp * 100)) : 0;
        fill.style.width = pct + '%';
        label.textContent = (hp || 0) + ' / ' + (maxHp || 0) + ' HP';
      }}
      setHp('my-hp-fill', 'my-hp-label', s.my_hp, s.my_max_hp);
      setHp('opp-hp-fill', 'opp-hp-label', s.opp_hp, s.opp_max_hp);

      document.getElementById('special-atk-count').textContent =
        (s.my_special_attack_uses == null ? '-' : s.my_special_attack_uses);
      document.getElementById('special-def-count').textContent =
        (s.my_special_defense_uses == null ? '-' : s.my_special_defense_uses);
    }});

    setInterval(() => {{
      socket.emit('request_state');
    }}, 400);

    // Pokémon list rendering & search
    const selected = {{ name: null }};
    function renderPokemonList(query) {{
      const list = document.getElementById('pokemon-list');
      list.innerHTML = '';
      const q = (query || '').toLowerCase();
      const pokes = (window.__POKEDEX || []).filter(p => {{
        return p.name.toLowerCase().includes(q) ||
               (p.type1 || '').toLowerCase().includes(q) ||
               (p.type2 || '').toLowerCase().includes(q);
      }});
      pokes.forEach(p => {{
        const div = document.createElement('div');
        div.className = 'pokemon-item' + (selected.name === p.name ? ' selected' : '');
        div.innerHTML = `
          <span>${{p.name}}</span>
          <span>
            <span class="badge">${{p.type1}}</span>
            ${{
              p.type2 ? '<span class="badge">'+p.type2+'</span>' : ''
            }}
          </span>`;
        div.onclick = () => {{
          selected.name = p.name;
          document.getElementById('selected-pokemon-label').textContent = p.name;
          renderPokemonList(document.getElementById('search').value);
          socket.emit('choose_pokemon', {{ name: p.name }});
        }};
        list.appendChild(div);
      }});
    }}
    document.getElementById('search').addEventListener('input', (e) => {{
      renderPokemonList(e.target.value);
    }});

    // Battle setup
    document.getElementById('btn-setup').addEventListener('click', () => {{
      socket.emit('send_battle_setup');
    }});

    // Attack + boosts
    let specialAtk = false;
    let specialDef = false;
    function updateAttackButtonGlow() {{
      const btnAtk = document.getElementById('btn-attack');
      if (specialAtk || specialDef) {{
        btnAtk.classList.add('glow');
      }} else {{
        btnAtk.classList.remove('glow');
      }}
    }}
    document.getElementById('btn-special-atk').addEventListener('click', () => {{
      specialAtk = !specialAtk;
      const btn = document.getElementById('btn-special-atk');
      if (specialAtk) btn.classList.add('toggled'); else btn.classList.remove('toggled');
      updateAttackButtonGlow();
    }});
    document.getElementById('btn-special-def').addEventListener('click', () => {{
      specialDef = !specialDef;
      const btn = document.getElementById('btn-special-def');
      if (specialDef) btn.classList.add('toggled'); else btn.classList.remove('toggled');
      updateAttackButtonGlow();
    }});
    document.getElementById('btn-attack').addEventListener('click', () => {{
      const move = document.getElementById('move-name').value || 'Attack';
      socket.emit('attack', {{
        move_name: move,
        special_attack: specialAtk,
        special_defense: specialDef
      }});
      specialAtk = false;
      specialDef = false;
      document.getElementById('btn-special-atk').classList.remove('toggled');
      document.getElementById('btn-special-def').classList.remove('toggled');
      updateAttackButtonGlow();
    }});

    // Chat
    document.getElementById('btn-send-chat').addEventListener('click', () => {{
      const input = document.getElementById('chat-input');
      const txt = input.value.trim();
      if (!txt) return;
      socket.emit('chat_text', {{ text: txt }});
      input.value = '';
    }});
    document.getElementById('chat-input').addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') {{
        document.getElementById('btn-send-chat').click();
      }}
    }});

    // Stickers: validate 320x320 and size <10MB
    document.getElementById('btn-send-sticker').addEventListener('click', () => {{
      const input = document.getElementById('sticker-input');
      const file = input.files && input.files[0];
      if (!file) {{
        alert('Choose an image first.');
        return;
      }}
      if (file.size >= 10 * 1024 * 1024) {{
        alert('Sticker must be smaller than 10MB.');
        return;
      }}
      const img = new Image();
      const reader = new FileReader();
      reader.onload = function(e) {{
        img.onload = function() {{
          if (img.width !== 320 || img.height !== 320) {{
            alert('Sticker must be exactly 320x320 pixels.');
            return;
          }}
          const base64 = e.target.result.split(',')[1];
          socket.emit('chat_sticker', {{ sticker_b64: base64 }});
        }};
        img.src = e.target.result;
      }};
      reader.readAsDataURL(file);
    }});
  </script>
</body>
</html>"""
        return html

    @app.route("/pokemon")
    def pokemon_endpoint():
        out = []
        for p in pokedex.values():
            out.append(
                {
                    "name": p.name,
                    "type1": p.type1,
                    "type2": p.type2,
                    "hp": p.hp,
                    "attack": p.attack,
                    "sp_attack": p.sp_attack,
                    "defense": p.defense,
                    "sp_defense": p.sp_defense,
                }
            )
        out.sort(key=lambda x: x["name"].lower())
        return jsonify({"pokemon": out})

    # SocketIO events

    @socketio.on("request_state")
    def _on_request_state():
        emit("state", peer.get_state_summary())

    @socketio.on("choose_pokemon")
    def _on_choose_pokemon(payload):
        if peer.role == Role.SPECTATOR:
            gui_debug("[GUI] Spectators cannot choose Pokémon")
            return
        name = (payload or {}).get("name", "")
        p = pokedex.get(name.lower())
        if not p:
            gui_debug(f"[GUI] Pokémon not found: {name}")
            return
        peer.my_side = BattleSide(pokemon=p, current_hp=p.hp)
        gui_debug(f"[GUI] Selected Pokémon: {p.name}")

    @socketio.on("send_battle_setup")
    def _on_send_battle_setup():
        if peer.role == Role.SPECTATOR:
            gui_debug("[GUI] Spectators cannot send battle setup")
            return
        peer.send_battle_setup()
        gui_debug("[GUI] BATTLE_SETUP sent")

    @socketio.on("attack")
    def _on_attack(payload):
        if peer.role == Role.SPECTATOR:
            gui_debug("[GUI] Spectators cannot attack")
            return
        move_name = (payload or {}).get("move_name", "Attack")
        sa = bool((payload or {}).get("special_attack", False))
        sd = bool((payload or {}).get("special_defense", False))
        peer.announce_attack(move_name, sa, sd)

    @socketio.on("chat_text")
    def _on_chat_text(payload):
        txt = (payload or {}).get("text", "")
        if not txt:
            return
        peer.send_chat_text(display_name, txt)

    @socketio.on("chat_sticker")
    def _on_chat_sticker(payload):
        b64 = (payload or {}).get("sticker_b64", "")
        if not b64:
            return
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception:
            gui_debug("[GUI] Invalid base64 sticker from browser.")
            return
        peer.send_chat_sticker(display_name, raw)

    if not peer.running:
        peer.start()

    gui_debug(f"[GUI] Browser GUI running on http://localhost:{http_port}")
    socketio.run(app, host="0.0.0.0", port=http_port)


# ===================== CLI entrypoint =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PokeProtocol peer with browser GUI.")
    parser.add_argument("--role", choices=["host", "joiner", "spectator"], required=True)
    parser.add_argument("--udp-port", type=int, default=50010)
    parser.add_argument("--host-ip", type=str, help="Host IP (for joiner/spectator)")
    parser.add_argument("--host-port", type=int, help="Host UDP port (for joiner/spectator)")
    parser.add_argument("--http-port", type=int, help="If set, run browser GUI on this port")
    parser.add_argument("--name", type=str, default="Player")
    args = parser.parse_args()

    role = Role(args.role)
    pokedex = load_pokemon_csv()

    if role == Role.HOST:
        peer = PokePeer(
            role=role,
            udp_port=args.udp_port,
            remote_host=None,
            remote_port=None,
            seed=12345,
        )
        peer.debug(f"[HOST] Listening on UDP port {args.udp_port}")
        peer.start()
    else:
        if not args.host_ip or not args.host_port:
            raise SystemExit("Joiner/Spectator must specify --host-ip and --host-port")
        peer = PokePeer(
            role=role,
            udp_port=args.udp_port,
            remote_host=args.host_ip,
            remote_port=args.host_port,
        )
        peer.start()
        peer.send_handshake_request()
        peer.debug(
            f"[{role.value.upper()}] Sent handshake request to {args.host_ip}:{args.host_port}"
        )

    if args.http_port:
        run_gui(peer, pokedex, http_port=args.http_port, display_name=args.name)
    else:
        peer.debug("[INFO] No GUI requested; running terminal-only.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            peer.stop()
