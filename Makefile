clean:
	rm -rf *~ dist *.egg-info build target

build-pyo3:
	maturin build -i python3 --features python -m rcube/Cargo.toml

develop-pyo3:
	maturin develop --release --features python -m rcube/Cargo.toml
