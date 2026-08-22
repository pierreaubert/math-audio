# --------------------------------------------------------- -*- just -*-
# How to install Just?
#	  cargo install just
# ----------------------------------------------------------------------

import 'builds/cross.just'

default:
	just --list

# ----------------------------------------------------------------------
# TEST
# ----------------------------------------------------------------------

test:
	cargo check --workspace --all-targets
	cargo test --workspace --lib --release

ntest:
	cargo nextest run --release --no-fail-fast --workspace --lib

# ----------------------------------------------------------------------
# FORMAT & LINT
# ----------------------------------------------------------------------

alias format := fmt

fmt:
	cargo fmt --all


lint:
	cargo clippy --all --features plotly -- -D warnings


# ----------------------------------------------------------------------
# PROD
# ----------------------------------------------------------------------

alias build := prod

prod: prod-workspace
	cargo build --release --bin plot-functions -p math-test-functions --features plotly
	cargo build --release --bin plot-de -p math-optimisation --features plotly
	cargo build --release --bin run-de -p math-optimisation
	cargo build --release --bin wav2csv -p math-dsp
	cargo build --release --bin simd-fuzzer -p math-dsp
	cargo build --release --bin benchmark-convergence -p math-optimisation

prod-workspace:
	cargo build --release --workspace

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

bench: bench-math

bench-math:
	cargo run --release --bin filter_bench -p math-iir-fir
	cargo bench -p math-autodiff --bench biquad_bench

# ----------------------------------------------------------------------
# CLEAN
# ----------------------------------------------------------------------

clean:
	cargo clean
	rm -f *.log *.wav *.json TAGS
	find . -name '*~' -exec rm {} \; -print
	find . -name 'Cargo.lock' -exec rm {} \; -print

# ----------------------------------------------------------------------
# DEV
# ----------------------------------------------------------------------

dev:
	cargo build --workspace

# ----------------------------------------------------------------------
# UPDATE
# ----------------------------------------------------------------------

update: update-rust update-pre-commit

update-rust:
	rustup update
	cargo update

update-pre-commit:
	pre-commit autoupdate

# ----------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------

examples: examples-math

examples-math: examples-autodiff examples-iir examples-optimisation examples-testfunctions

examples-autodiff:
	cargo run --release --example biquad_match -p math-autodiff
	cargo run --release --example fdn_direct_match -p math-autodiff
	cargo run --release --example fdn_match -p math-autodiff
	cargo run --release --example geq_match -p math-autodiff
	cargo run --release --example peq_match -p math-autodiff
	cargo run --release --example svf_match -p math-autodiff

examples-iir:
	cargo run --release --example format_demo -p math-iir-fir
	cargo run --release --example format_rme_room_demo -p math-iir-fir
	cargo run --release --example readme_example -p math-iir-fir
	cargo run --release --example fir_example -p math-iir-fir
	cargo run --release --example peq_loudness_compensation -p math-iir-fir

examples-optimisation:
	cargo run --release --example optde_basic -p math-optimisation
	cargo run --release --example optde_adaptive_demo -p math-optimisation
	cargo run --release --example optde_linear_constraints -p math-optimisation
	cargo run --release --example optde_nonlinear_constraints -p math-optimisation
	cargo run --release --example optde_parallel -p math-optimisation

examples-testfunctions:
	cargo run --release --example test_hartman_4d -p math-test-functions
	cargo run --release --example test_new_sfu_functions -p math-test-functions

# ----------------------------------------------------------------------
# Install rustup
# ----------------------------------------------------------------------

install-rustup:
	curl https://sh.rustup.rs -sSf > ./scripts/install-rustup
	chmod +x ./scripts/install-rustup
	./scripts/install-rustup -y
	~/.cargo/bin/rustup default stable
	~/.cargo/bin/cargo install just
	~/.cargo/bin/cargo install cargo-wizard
	~/.cargo/bin/cargo install cargo-llvm-cov
	~/.cargo/bin/cargo install cargo-bininstall
	~/.cargo/bin/cargo binstall cargo-nextest --secure

# ----------------------------------------------------------------------
# Install macos
# ----------------------------------------------------------------------

install-macos-cross:
	# use git version until 0.2.6 is out
	cargo install cross --git https://github.com/cross-rs/cross
	cross target add x86_64-apple-ios

install-macos-brew:
	curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh > ./scripts/install-brew
	chmod +x ./scripts/install-brew
	NONINTERACTIVE=1 ./scripts/install-brew

install-macos: install-macos-brew install-rustup
	# need xcode
	xcode-select --install
	# chromedriver sheanigans
	brew install chromedriver
	xattr -d com.apple.quarantine $(which chromedriver)
	# optimisation library
	brew install nlopt cmake


# ----------------------------------------------------------------------
# Install linux
# ----------------------------------------------------------------------

install-linux-root:
	sudo apt update && sudo apt -y install \
	   perl curl build-essential gcc g++ pkg-config cmake ninja-build gfortran \
	   libssl-dev \
	   ca-certificates \
	   patchelf libopenblas-dev gfortran \
	   chromium-browser chromium-chromedriver

install-linux: install-linux-root install-rustup

install-ubuntu-common:
		sudo apt install -y \
			 curl \
			 build-essential gcc g++ \
			 pkg-config \
			 libssl-dev \
			 ca-certificates \
			 cmake \
			 ninja-build \
			 perl \
			 patchelf \
			 libopenblas-dev \
			 gfortran

install-ubuntu-x86-driver :
		sudo apt install -y \
			 chromium-browser \
			 chromium-chromedriver

install-ubuntu-arm64-driver :
		sudo apt install -y firefox
		# where is the geckodriver ?

install-ubuntu-x86: install-ubuntu-common install-ubuntu-x86-driver

install-ubuntu-arm64: install-ubuntu-common install-ubuntu-arm64-driver


# ----------------------------------------------------------------------
# publish
# ----------------------------------------------------------------------

publish: publish-math

publish-math:
	cd crates/math-test-functions && cargo publish
	cd crates/math-optimisation && cargo publish
	cd crates/math-iir-fir && cargo publish
	cd crates/math-dsp && cargo publish
	cd crates/math-rir && cargo publish
	cd crates/math-delaunay && cargo publish
	cd crates/math-convex-hull && cargo publish
	cd crates/math-autodiff && cargo publish

# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------

# Per-crate coverage ratchet thresholds (lines %, measured 2026-08-18,
# integration tests included). Raise when coverage improves; never lower.
# dsp/optimisation include their (largely untested) bin targets, hence the
# lower values versus the old --lib-only measurement.
qa_cov_analog := "78"
qa_cov_autodiff := "76"
qa_cov_convex_hull := "90"
qa_cov_delaunay := "88"
qa_cov_dsp := "82"
qa_cov_iir_fir := "80"
qa_cov_optimisation := "62"
qa_cov_rir := "94"
qa_cov_test_functions := "90"

# plotly's askama templates fail to compile when the cargo registry path
# traverses a symlinked ~/.cargo; use the canonical path everywhere.
# (export applies to the whole Justfile regardless of where it appears.)
export CARGO_HOME := env_var_or_default("CARGO_HOME", canonicalize(home_directory() / ".cargo"))

[private]
_qa crate threshold:
	echo "==================== QA: {{crate}} ===================="
	cargo fmt -p {{crate}} -- --check
	cargo clippy -p {{crate}} --all-targets -- -D warnings
	cargo test -p {{crate}} --lib --release
	cargo test -p {{crate}} --tests --release
	cargo test -p {{crate}} --doc
	cargo bench -p {{crate}} --no-run
	cargo llvm-cov -p {{crate}} --summary-only --release --fail-under-lines {{threshold}}

qa-convex-hull: (_qa "math-convex-hull" qa_cov_convex_hull)

qa-delaunay: (_qa "math-delaunay" qa_cov_delaunay)

qa-rir: (_qa "math-rir" qa_cov_rir)
	cargo bench -p math-rir --bench iso3382 -- --quick

qa-test-functions: (_qa "math-test-functions" qa_cov_test_functions) examples-testfunctions
	cargo run --release -p math-test-functions --example test_additional_functions
	cargo run --release -p math-test-functions --example test_gramacy_lee
	cargo run --release -p math-test-functions --example find_hartman_4d_min
	cargo build --release --bin plot-functions -p math-test-functions --features plotly
	cargo bench -p math-test-functions --bench eval -- --quick

qa-iir-fir: (_qa "math-iir-fir" qa_cov_iir_fir) examples-iir
	cargo build --release --bin filter_bench -p math-iir-fir
	cargo bench -p math-iir-fir --bench biquad_bench -- --quick
	cargo bench -p math-iir-fir --bench response_bench -- --quick
	cargo bench -p math-iir-fir --bench fir_design_bench -- --quick

qa-optimisation: (_qa "math-optimisation" qa_cov_optimisation) examples-optimisation
	cargo build --release --bin run-de -p math-optimisation
	cargo build --release --bin benchmark-convergence -p math-optimisation
	cargo build --release --bin plot-de -p math-optimisation --features plotly
	cargo bench -p math-optimisation --bench de_bench -- --quick
	cargo bench -p math-optimisation --bench cmaes_bench -- --quick

qa-autodiff: (_qa "math-autodiff" qa_cov_autodiff) examples-autodiff
	cargo bench -p math-autodiff --bench biquad_bench -- --quick

qa-dsp: (_qa "math-dsp" qa_cov_dsp)
	cargo run --release --bin simd-fuzzer -p math-dsp
	cargo build --release --bin wav2csv -p math-dsp
	if [ "$(uname -m)" = "x86_64" ]; then RUSTFLAGS="-C target-feature=+avx2" cargo check -p math-dsp --all-targets && RUSTFLAGS="-C target-feature=+avx2" cargo test -p math-dsp; else echo "Skipping AVX2 pass (not x86_64)"; fi
	cargo bench -p math-dsp --bench welch_spectrum -- --quick
	cargo bench -p math-dsp --bench audio_features -- --quick

qa-analog: (_qa "math-analog" qa_cov_analog)
	cargo run --release -p math-analog --example analysis_report
	cargo run --release -p math-analog --example alias_reference_report
	cargo run --release -p math-analog --example console_preamp_report
	cargo run --release -p math-analog --example model_matrix_report
	cargo run --release -p math-analog --example fitting_report --features fitting
	cargo bench -p math-analog --bench harmonics -- --quick

alias qa-math := qa

qa: qa-analog qa-autodiff qa-convex-hull qa-delaunay qa-dsp qa-iir-fir qa-optimisation qa-rir qa-test-functions
	cargo clippy --all --features plotly -- -D warnings
	cargo llvm-cov --summary-only --release --fail-under-lines 90

# ----------------------------------------------------------------------
# POST
# ----------------------------------------------------------------------

post-install:
	$HOME/.cargo/bin/rustup default stable
	$HOME/.cargo/bin/cargo install just
	$HOME/.cargo/bin/cargo check
