import { describe, expect, it } from "vitest";
import { MOVES, computerMove, isMove, judge, newGame, playRound } from "../src/game.js";

describe("judge", () => {
  it.each([
    ["rock", "scissors", "win"],
    ["paper", "rock", "win"],
    ["scissors", "paper", "win"],
    ["scissors", "rock", "lose"],
    ["rock", "paper", "lose"],
    ["paper", "scissors", "lose"],
  ])("%s vs %s -> %s", (p, c, out) => expect(judge(p, c)).toBe(out));

  it("draws on equal moves", () => MOVES.forEach((m) => expect(judge(m, m)).toBe("draw")));
  it("rejects unknown moves", () => expect(() => judge("lizard", "rock")).toThrow());
});

describe("computerMove", () => {
  it("maps rng ranges to each move, including the edge 1 - epsilon", () => {
    expect(computerMove(() => 0)).toBe("rock");
    expect(computerMove(() => 0.34)).toBe("paper");
    expect(computerMove(() => 0.9999999)).toBe("scissors");
    expect(computerMove(() => 1)).toBe("scissors"); // a bad rng must not return undefined
  });
});

describe("playRound", () => {
  it("updates score and history without mutating the input", () => {
    const s0 = newGame();
    const s1 = playRound(s0, "rock", () => 0.9); // computer: scissors
    const s2 = playRound(s1, "rock", () => 0.5); // computer: paper
    const s3 = playRound(s2, "rock", () => 0); // computer: rock
    expect(s0).toEqual(newGame());
    expect(s3.score).toEqual({ player: 1, computer: 1, draws: 1 });
    expect(s3.rounds.map((r) => r.outcome)).toEqual(["win", "lose", "draw"]);
  });
  it("throws on an invalid player move", () =>
    expect(() => playRound(newGame(), "spock")).toThrow());
});

it("isMove", () => {
  expect(isMove("paper")).toBe(true);
  expect(isMove("Paper")).toBe(false);
});
