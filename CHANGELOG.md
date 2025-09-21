# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-20

### Added
- Initial release of FrankensteinDB
- Core WebsiteDNA compression system with ~1KB fingerprints
- ScyllaDB emulator for time-series website evolution tracking
- Redis context store with graceful memory fallback
- SQLite FTS5 search index for website discovery
- Blob storage system for cached web content
- Unified async API for web intelligence analytics
- Basic usage example demonstrating core functionality
- Advanced web scraping example with requests/BeautifulSoup
- Comprehensive documentation and setup instructions
- MIT license for open source distribution
- Full test coverage for core components

### Features
- **WebsiteDNA Compression**: LZMA + MessagePack compression achieving ~1KB website fingerprints
- **Time-Series Storage**: SQLite-based emulation of ScyllaDB for tracking website evolution
- **Context Management**: Redis-backed user context store with automatic memory fallback
- **Full-Text Search**: SQLite FTS5 engine for fast website content discovery
- **Blob Storage**: Efficient file-based storage with compression and cleanup
- **Analytics**: Comprehensive website intelligence and framework detection
- **Async Architecture**: Full async/await support with ThreadPoolExecutor
- **Graceful Degradation**: Works without Redis, handles errors elegantly
- **Framework Detection**: Automatic detection of web frameworks and technologies
- **Evolution Tracking**: Track and analyze website changes over time

### Documentation
- Complete README with installation and usage instructions
- Inline code documentation and docstrings
- Example scripts for basic usage and web scraping
- Requirements file with optional dependencies
- MIT license for commercial and open source use