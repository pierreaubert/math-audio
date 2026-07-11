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

prod-workspace:
	cargo build --release --workspace

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

bench: bench-math

bench-math:
	cargo run --release --bin benchmark-convergence -p math-optimisation
	cargo run --release --bin biquad-bench -p math-iir-fir
	cargo bench --release -p math-autodiff --bench biquad_bench

# ----------------------------------------------------------------------
# CLEAN
# ----------------------------------------------------------------------

clean:
	cargo clean
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

qa: qa-math

qa-math:
	cargo run --release --bin simd-fuzzer -p math-dsp

# ----------------------------------------------------------------------
# POST
# ----------------------------------------------------------------------

post-install:
	$HOME/.cargo/bin/rustup default stable
	$HOME/.cargo/bin/cargo install just
	$HOME/.cargo/bin/cargo check
