# Development Guide for AWS DLC MCP Server

This guide provides instructions for setting up a development environment for the AWS Deep Learning Containers MCP Server.

## Prerequisites

- Python 3.10 or higher
- Docker
- AWS CLI configured with appropriate permissions
- Git

## Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/awslabs/mcp.git
   cd mcp/src/dlc-mcp-server
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=awslabs.dlc_mcp_server
```

## Code Quality

Format code with Black:

```bash
black awslabs
```

Sort imports with isort:

```bash
isort awslabs
```

Run linting with Ruff:

```bash
ruff check awslabs
```

Run type checking with mypy:

```bash
mypy awslabs
```

## Running the Server Locally

Start the server in development mode:

```bash
FASTMCP_LOG_LEVEL=DEBUG dlc-mcp-server
```

To enable write operations:

```bash
ALLOW_WRITE=true FASTMCP_LOG_LEVEL=DEBUG dlc-mcp-server
```

## Project Structure

- `awslabs/dlc_mcp_server/`: Main package
  - `api/`: API definitions and schemas
  - `modules/`: Functional modules (image building, deployment, etc.)
  - `templates/`: Jinja2 templates for Dockerfiles and configurations
  - `utils/`: Utility functions and helpers
  - `main.py`: Server entry point

## Adding New Features

1. Create a new module in `awslabs/dlc_mcp_server/modules/`
2. Implement the necessary functions and classes
3. Register the module in `main.py`
4. Add tests in the `tests/` directory

## Documentation

Update the README.md file with any new features or changes.

## Release Process

1. Update the version in `pyproject.toml`
2. Create a new tag
3. Push the tag to trigger the release workflow

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
