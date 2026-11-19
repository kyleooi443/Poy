/** Pure rock-paper-scissors rules. No DOM, no randomness of its own (pass `rng`). */

/** @typedef {"rock" | "paper" | "scissors"} Move */
/** @typedef {"win" | "lose" | "draw"} Outcome */
/** @typedef {{ player: number, computer: number, draws: number }} Score */
/** @typedef {{ player: Move, computer: Move, outcome: Outcome }} Round */
/** @typedef {{ score: Score, rounds: Round[] }} GameState */

/** @type {readonly Move[]} */
export const MOVES = Object.freeze(["rock", "paper", "scissors"]);

/** @type {Readonly<Record<Move, Move>>} what each move beats */
const BEATS = Object.freeze({ rock: "scissors", paper: "rock", scissors: "paper" });

/** @param {unknown} x @returns {x is Move} */
export const isMove = (x) => MOVES.includes(/** @type {Move} */ (x));

/**
 * Outcome from the player's point of view.
 * @param {Move} player @param {Move} computer @returns {Outcome}
 */
export function judge(player, computer) {
  if (!isMove(player) || !isMove(computer))
    throw new Error(`invalid move: ${player} vs ${computer}`);
  if (player === computer) return "draw";
  return BEATS[player] === computer ? "win" : "lose";
}

/** @param {() => number} rng returns [0, 1) @returns {Move} */
export function computerMove(rng = Math.random) {
  return MOVES[Math.min(MOVES.length - 1, Math.floor(rng() * MOVES.length))];
}

/** @returns {GameState} */
export const newGame = () => ({ score: { player: 0, computer: 0, draws: 0 }, rounds: [] });

/**
 * Play one round. Returns a new state; the input is not modified.
 * @param {GameState} state @param {Move} player @param {() => number} [rng]
 * @returns {GameState}
 */
export function playRound(state, player, rng = Math.random) {
  const computer = computerMove(rng);
  const outcome = judge(player, computer);
  const s = state.score;
  return {
    score: {
      player: s.player + (outcome === "win" ? 1 : 0),
      computer: s.computer + (outcome === "lose" ? 1 : 0),
      draws: s.draws + (outcome === "draw" ? 1 : 0),
    },
    rounds: [...state.rounds, { player, computer, outcome }],
  };
}
