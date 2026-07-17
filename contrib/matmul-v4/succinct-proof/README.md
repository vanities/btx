# Succinct sketch proof: decoupling the v4 proof size from m

Reference implementation + tests for replacing the raw on-chain sketch payload
with a constant-or-near-constant-size proof. Target problem: in the current v4
design the block carries the full m x m sketch (8 * m^2 bytes: 8 MiB at profile
C, 32 MiB at profile D), so the proof size and the projection dimension m are
conjoined. This scheme cuts that link: m can grow (better compute-scaling) while
the on-chain proof stays small and (for KZG) literally constant.

## The idea in one paragraph

The verifier never needs the sketch matrix itself. It needs to be convinced that
the committed sketch equals U*A*B*V. Freivalds already reduces that to comparing
two scalars. So: commit to Chat as the coefficients of a bivariate polynomial
f(X,Y) = sum_ij Chat_ij X^i Y^j (32-byte commitment), derive a challenge point
(a,b) by Fiat-Shamir from the header and the commitment, and have the miner
prove the single evaluation f(a,b) with a polynomial-commitment opening. The
verifier recomputes the same scalar from the seed on its own:
u_a^T (U (A (B (V v_b)))) with u_a = (a^i), v_b = (b^j), pure matrix-vector
work, O(n^2), never forming any matrix product. Equality at a random point
implies (whp) equality of the polynomials, i.e. the committed sketch is the
real one. On-chain data: commitment + one field element + one opening proof.
Independent of m.

## Soundness

If the committed Chat is wrong, (f - true)(X,Y) is a nonzero polynomial of
degree < m in each variable. By Schwartz-Zippel a random (a,b) in F_q^2
satisfies it with probability <= 2(m-1)/q. With q = 2^61 - 1 and m = 2048 that
is about 2^-49 per round. Fiat-Shamir binds (a,b) to the commitment, so the
prover commits first and learns the challenge second. Grinding interplay: each
grind attempt costs a full commit over the sketch, so a 2^-49-per-try cheat is
already worse than honest mining; two independent rounds (or one round with
(a,b) drawn from a degree-2 extension field) push it to ~2^-98, out of reach of
any physical grinding budget. The compute requirement of the PoW is untouched:
the miner must still produce the real Chat to commit to it.

## What the miner pays

- Per candidate nonce: one commitment over the sketch. The current design
  already hashes the full serialized sketch per candidate (the digest
  H(sigma || Chat)), and a Merkle-style commitment over the coefficients is
  about 2x that hashing. So the per-nonce cost class does not change.
- One flavor detail to settle in production: a textbook FRI opening commits to
  the low-degree-extended codeword (blowup 8), which per nonce would add an NTT
  plus 8x the hashing. Still small next to the matmul itself, but the cleaner
  design is a coefficient commitment per nonce with the LDE built only for the
  won block, which requires the PCS flavor that supports opening against a
  coefficient root. This is the main open engineering question.
- Per WON block only: one opening-proof generation (FRI: seconds-class on the
  same hardware; amortized once per block, not per nonce). Note this adds proof
  latency at block-publish time, which trades against orphan risk; pipelining
  or header-first relay with the proof following are the standard mitigations.
- This argues for a transparent hash-based commitment (FRI/Merkle) over KZG for
  the mining side: KZG commitment per nonce would be a multi-million-point MSM,
  which is not viable per candidate. KZG numbers are included below only to
  show the constant-size endpoint.

## Sizes (measured by this reference's size models)

| n | m | naive raw sketch (8m^2) | succinct FRI (upper est.) | succinct KZG |
|------|------|------------------------|---------------------------|--------------|
| 1024 | 256 | 512 KiB | ~169 KiB | 168 B |
| 2048 | 512 | 2 MiB | ~213 KiB | 168 B |
| 4096 | 1024 (profile C) | 8 MiB | ~261 KiB | 168 B |
| 4096 | 2048 (profile D) | 32 MiB | ~314 KiB | 168 B |

The FRI column is a conservative upper estimate (34 queries, blowup 8, no path
deduplication); production implementations typically land 3-10x smaller. The
headline: naive grows 4x per m-doubling, FRI grows ~log^2, KZG does not move.
At profile D the shrink is >100x even under the conservative estimate, and the
permanent chain no longer contains any m-sized object at all.

## What this reference is, and is not

- IS: a faithful, tested model of the protocol. Operand generation from seed,
  the honest O(n^3) sketch, the bilinear-to-polynomial reduction, Fiat-Shamir,
  prover, verifier, tamper detection, and the size accounting.
- IS NOT: production cryptography. The polynomial commitment is modeled as an
  ideal functionality (`IdealPC`): binding and evaluation-correctness are
  enforced in code, and proof sizes are reported via standard FRI/KZG formulas.
  Swapping `IdealPC` for a real FRI opening (transparent, hash-only, no trusted
  setup, the recommended flavor for a PoW chain) yields a production prototype.
  The reduction and the soundness argument are unchanged by that swap.

## Relation to the segregated-proof design

Complementary, not competing. Segregation + pruning (the existing design doc)
relocates the raw sketch off the permanent chain and is the right near-term
ship: standard pattern, no new cryptography. This scheme is the endgame: the
sketch never exists on-chain in any form, archival nodes carry nothing m-sized,
block relay carries kilobytes instead of megabytes, and the C-vs-D choice stops
having any chain-size consequence at all. A natural path is to ship segregation
first and introduce the succinct proof as a later upgrade that reuses the same
header commitment slot (matmul_digest becomes the polynomial commitment).

## Adversarial cost

Can a miner pass while doing less compute? `adversarial_cost.py` prices every
known strategy and the tests assert none beats honest mining. Partial compute
(garbage-fill a fraction e of entries) loses exponentially: Q=34 sampled
openings accept with (1-e)^Q while the savings are linear, so skipping 1% of
the sketch already costs 1.4x honest per accepted block, 10% costs ~35x, and
the curve never turns profitable. The acceptance model is verified against an
actual Merkle commitment with root-derived query sampling, not just the
formula. Garbage-commit scanning is Schwartz-Zippel-priced at ~2^-49 per
round against a nonce-throughput speedup bounded by ~2^30, and challenge
grinding costs ~10^11 honest blocks per success. Two implementation MUSTs
fall out: the commitment stays in the header hash as the per-nonce
eligibility gate, and the PCS admits no free re-randomization.

## Run

```
python3 succinct_matmul_pow.py   # self-check + decoupling table
python3 adversarial_cost.py      # adversarial cost table (modeled + sampled)
python3 test_succinct.py         # 23 tests: completeness, soundness, binding,
                                 # decoupling, verifier-cost, adversarial cost
```

No dependencies beyond the Python standard library.
