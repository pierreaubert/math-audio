# Changelog

## 0.17.0

- Upgrade ZIP extraction to ZIP 8 with the dependency's default features retained.
- `InstallationError::ExtractZip` now carries ZIP 8's `ZipError`; this is a public nominal-type change.
