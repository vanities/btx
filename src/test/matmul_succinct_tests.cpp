// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
//
// Succinct sketch-proof reduction tests (chain-size decoupling;
// contrib/matmul-v4/succinct-proof/README.md). This suite exercises the
// reduction on the REAL v4.2 consensus pipeline (ExpandOperand* /
// ComputeProjected* / ComputeCombineModQ / ComputeSketchDigest), with the
// commitment role played by the existing matmul_digest = H(sigma || Chat):
//
//   (a) IDENTITY: for the honest sketch, the committed-polynomial evaluation
//       f(a,b) = sum_ij Chat_ij a^i b^j equals the verifier-side scalar
//       u_a^T (U (A (B (V v_b)))) computed from the seed with matrix-vector
//       products only, O(n^2), at a Fiat-Shamir challenge bound to the digest.
//   (b) SOUNDNESS: a single-entry perturbation of Chat, re-committed so it is
//       digest-consistent, breaks the identity at its own challenge point
//       (the Schwartz-Zippel catch: <= 2(m-1)/q per round).
//   (c) BINDING: a different committed sketch yields a different digest and a
//       different Fiat-Shamir challenge.
//   (d) DETERMINISM: run-to-run identity of digest, challenge, and both
//       scalars.
//   (e) ROBUSTNESS: the identity holds at an independent second challenge
//       point, so (a) is structural rather than coincidental.
//
// The polynomial-commitment OPENING (FRI or similar) is deliberately out of
// scope here: this suite validates the reduction the opening would attest to.
// The verifier path below never forms a matrix product and never reads the
// sketch except through the claimed polynomial evaluation.

#include <matmul/int8_field.h>
#include <matmul/matmul_v4.h>
#include <matmul/matmul_v4_bmx4.h>
#include <matmul/pow_v4.h>

#include <crypto/sha256.h>
#include <primitives/block.h>
#include <test/util/setup_common.h>
#include <uint256.h>

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

using namespace matmul::v4;
namespace bx = matmul::v4::bmx4;
namespace f8 = matmul::int8_field;
using f8::Fq;

BOOST_FIXTURE_TEST_SUITE(matmul_succinct_tests, BasicTestingSetup)

namespace {

constexpr uint32_t kTestDim = 256; // fast unit dimension (b=4 -> m=64)

uint256 ParseUint256(std::string_view hex)
{
    const auto parsed = uint256::FromHex(hex);
    BOOST_REQUIRE(parsed.has_value());
    return *parsed;
}

CBlockHeader MakeV4Header(uint64_t nonce, uint32_t n)
{
    CBlockHeader header;
    header.nVersion = 0x20000004;
    header.hashPrevBlock = ParseUint256("5151515151515151515151515151515151515151515151515151515151515151");
    header.hashMerkleRoot = ParseUint256("a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3");
    header.nTime = 1'770'000'000;
    header.nBits = 0x207fffff;
    header.nNonce64 = nonce;
    header.nNonce = static_cast<uint32_t>(nonce);
    header.matmul_dim = static_cast<uint16_t>(n);
    header.seed_a = ParseUint256("1111111111111111111111111111111111111111111111111111111111111111");
    header.seed_b = ParseUint256("2222222222222222222222222222222222222222222222222222222222222222");
    return header;
}

// Fiat-Shamir challenge component: SHA256(tag || commitment) mapped into
// F_q \ {0}. The commitment (matmul_digest) already binds header and sketch
// bytes (digest = H(sigma || payload), sigma = DeriveSigma(header)), so the
// challenge inherits both bindings.
Fq FsChallenge(const uint256& commitment, uint8_t tag)
{
    unsigned char out[CSHA256::OUTPUT_SIZE];
    CSHA256().Write(&tag, 1).Write(commitment.begin(), 32).Finalize(out);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | out[i];
    v %= f8::kFieldPrime;
    return v == 0 ? Fq{1} : static_cast<Fq>(v);
}

std::vector<Fq> Powers(Fq x, uint32_t count)
{
    std::vector<Fq> p(count);
    p[0] = 1;
    for (uint32_t i = 1; i < count; ++i) p[i] = f8::FqMul(p[i - 1], x);
    return p;
}

// Prover-side evaluation of the committed polynomial:
// f(a,b) = sum_ij Chat[i*m+j] a^i b^j. O(m^2); needs the sketch.
Fq EvalSketchPoly(const std::vector<Fq>& chat, uint32_t m, Fq a, Fq b)
{
    const std::vector<Fq> pa = Powers(a, m);
    const std::vector<Fq> pb = Powers(b, m);
    Fq acc = 0;
    for (uint32_t i = 0; i < m; ++i) {
        Fq row = 0;
        for (uint32_t j = 0; j < m; ++j) {
            row = f8::FqAdd(row, f8::FqMul(chat[static_cast<size_t>(i) * m + j], pb[j]));
        }
        acc = f8::FqAdd(acc, f8::FqMul(pa[i], row));
    }
    return acc;
}

// s8 matrix (rows x cols, row-major) times Fq vector: out = M * v. O(rows*cols).
std::vector<Fq> MatVec(const std::vector<int8_t>& M, uint32_t rows, uint32_t cols,
                       const std::vector<Fq>& v)
{
    BOOST_REQUIRE_EQUAL(v.size(), cols);
    std::vector<Fq> out(rows, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        Fq acc = 0;
        const size_t base = static_cast<size_t>(r) * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            acc = f8::FqAdd(acc, f8::FqMul(f8::FqFromSigned(M[base + c]), v[c]));
        }
        out[r] = acc;
    }
    return out;
}

// Verifier-side scalar u_a^T (U (Ahat (Bhat (V v_b)))): matrix-vector products
// only, associated right to left. Never forms U*Ahat, Bhat*V, or the sketch.
Fq VerifierSideScalar(const std::vector<int8_t>& Ahat, const std::vector<int8_t>& Bhat,
                      const std::vector<int8_t>& U, const std::vector<int8_t>& V,
                      uint32_t n, uint32_t m, Fq a, Fq b)
{
    const std::vector<Fq> ua = Powers(a, m);
    const std::vector<Fq> vb = Powers(b, m);
    const std::vector<Fq> z = MatVec(V, n, m, vb);      // V v_b        (n)
    const std::vector<Fq> y = MatVec(Bhat, n, n, z);    // Bhat (V v_b) (n)
    const std::vector<Fq> x = MatVec(Ahat, n, n, y);    // Ahat ...     (n)
    const std::vector<Fq> t = MatVec(U, m, n, x);       // U ...        (m)
    Fq acc = 0;
    for (uint32_t i = 0; i < m; ++i) acc = f8::FqAdd(acc, f8::FqMul(ua[i], t[i]));
    return acc;
}

struct Pipeline {
    uint32_t m{0};
    std::vector<int8_t> Ahat, Bhat, U, V;
    std::vector<Fq> chat;
    uint256 sigma;
    uint256 digest; // = ComputeSketchDigest(sigma, SerializeSketch(chat))
};

// Run the REAL ENC-BMX4C pipeline for a header: operand expansion, projections,
// combine, and the consensus digest over the serialized sketch.
Pipeline RunPipeline(const CBlockHeader& header, uint32_t n)
{
    Pipeline p;
    BOOST_REQUIRE(bx::ValidateDimsBMX4C(n, kTileB, p.m));
    p.sigma = DeriveSigma(header);
    const uint256 seed_a = bx::DeriveOperandSeedBMX4C(header, Operand::A);
    const uint256 seed_b = bx::DeriveOperandSeedBMX4C(header, Operand::B);
    const auto [seed_u, seed_v] = bx::DeriveProjectorSeedsBMX4C(header);
    p.Ahat = bx::ExpandOperandA(seed_a, n);
    p.Bhat = bx::ExpandOperandB(seed_b, n);
    p.U = bx::ExpandProjectorBMX4C(seed_u, p.m, n);
    p.V = bx::ExpandProjectorBMX4C(seed_v, n, p.m);
    const std::vector<int32_t> P = ComputeProjectedLeft(p.U, p.Ahat, n, p.m);
    const std::vector<int32_t> Q = ComputeProjectedRight(p.Bhat, p.V, n, p.m);
    p.chat = ComputeCombineModQ(P, Q, n, p.m);
    p.digest = ComputeSketchDigest(p.sigma, SerializeSketch(p.chat));
    return p;
}

} // namespace

// --- (a) IDENTITY ------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduction_accepts_honest_sketch)
{
    const CBlockHeader header = MakeV4Header(/*nonce=*/1, kTestDim);
    const Pipeline p = RunPipeline(header, kTestDim);
    const Fq a = FsChallenge(p.digest, /*tag=*/0x61); // 'a'
    const Fq b = FsChallenge(p.digest, /*tag=*/0x62); // 'b'
    const Fq claimed = EvalSketchPoly(p.chat, p.m, a, b);
    const Fq recomputed = VerifierSideScalar(p.Ahat, p.Bhat, p.U, p.V, kTestDim, p.m, a, b);
    BOOST_CHECK_EQUAL(claimed, recomputed);
}

BOOST_AUTO_TEST_CASE(reduction_accepts_honest_sketch_second_nonce)
{
    const CBlockHeader header = MakeV4Header(/*nonce=*/2, kTestDim);
    const Pipeline p = RunPipeline(header, kTestDim);
    const Fq a = FsChallenge(p.digest, 0x61);
    const Fq b = FsChallenge(p.digest, 0x62);
    BOOST_CHECK_EQUAL(EvalSketchPoly(p.chat, p.m, a, b),
                      VerifierSideScalar(p.Ahat, p.Bhat, p.U, p.V, kTestDim, p.m, a, b));
}

// --- (b) SOUNDNESS -----------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduction_rejects_wrong_but_digest_consistent_sketch)
{
    const CBlockHeader header = MakeV4Header(/*nonce=*/1, kTestDim);
    const Pipeline p = RunPipeline(header, kTestDim);

    // Perturb one entry, then RE-COMMIT so the digest matches the tampered
    // sketch (a digest-consistent cheat, the strongest form: the hash check
    // alone cannot catch it; only the reduction can).
    std::vector<Fq> tampered = p.chat;
    tampered[0] = f8::FqAdd(tampered[0], 1);
    const uint256 digest2 = ComputeSketchDigest(p.sigma, SerializeSketch(tampered));
    const Fq a2 = FsChallenge(digest2, 0x61);
    const Fq b2 = FsChallenge(digest2, 0x62);

    const Fq claimed = EvalSketchPoly(tampered, p.m, a2, b2);
    const Fq recomputed = VerifierSideScalar(p.Ahat, p.Bhat, p.U, p.V, kTestDim, p.m, a2, b2);
    BOOST_CHECK(claimed != recomputed);
}

// --- (c) BINDING -------------------------------------------------------------

BOOST_AUTO_TEST_CASE(challenge_binds_to_commitment)
{
    const CBlockHeader header = MakeV4Header(/*nonce=*/1, kTestDim);
    const Pipeline p = RunPipeline(header, kTestDim);

    std::vector<Fq> tampered = p.chat;
    tampered[0] = f8::FqAdd(tampered[0], 1);
    const uint256 digest2 = ComputeSketchDigest(p.sigma, SerializeSketch(tampered));

    BOOST_CHECK(p.digest != digest2);
    BOOST_CHECK(FsChallenge(p.digest, 0x61) != FsChallenge(digest2, 0x61) ||
                FsChallenge(p.digest, 0x62) != FsChallenge(digest2, 0x62));
}

// --- (d) DETERMINISM ---------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduction_determinism_run_to_run)
{
    const CBlockHeader header = MakeV4Header(/*nonce=*/1, kTestDim);
    const Pipeline p1 = RunPipeline(header, kTestDim);
    const Pipeline p2 = RunPipeline(header, kTestDim);
    BOOST_CHECK(p1.digest == p2.digest);
    const Fq a = FsChallenge(p1.digest, 0x61);
    const Fq b = FsChallenge(p1.digest, 0x62);
    BOOST_CHECK_EQUAL(EvalSketchPoly(p1.chat, p1.m, a, b),
                      EvalSketchPoly(p2.chat, p2.m, a, b));
    BOOST_CHECK_EQUAL(VerifierSideScalar(p1.Ahat, p1.Bhat, p1.U, p1.V, kTestDim, p1.m, a, b),
                      VerifierSideScalar(p2.Ahat, p2.Bhat, p2.U, p2.V, kTestDim, p2.m, a, b));
}

// --- (e) ROBUSTNESS ----------------------------------------------------------

BOOST_AUTO_TEST_CASE(identity_holds_at_independent_second_challenge)
{
    // A second, independently tagged challenge point also satisfies the
    // identity: (a) is the polynomial identity, not a lucky point.
    const CBlockHeader header = MakeV4Header(/*nonce=*/1, kTestDim);
    const Pipeline p = RunPipeline(header, kTestDim);
    const Fq a = FsChallenge(p.digest, 0x63); // independent tags
    const Fq b = FsChallenge(p.digest, 0x64);
    BOOST_CHECK_EQUAL(EvalSketchPoly(p.chat, p.m, a, b),
                      VerifierSideScalar(p.Ahat, p.Bhat, p.U, p.V, kTestDim, p.m, a, b));
}

BOOST_AUTO_TEST_SUITE_END()
