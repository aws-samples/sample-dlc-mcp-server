# Development Guide for AWS DLC MCP Server

This guide provides instructions for setting up a development environment for the AWS Deep Learning Containers MCP Server.

## Prerequisites

- Python 3.10 or higher
- Docker
- AWS CLI configured with appropriate permissions
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/aws-samples/sample-dlc-mcp-server.git
   cd sample-dlc-mcp-server
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Running Tests

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run tests with coverage:

```bash
python -m pytest tests/ --cov=aws_samples.dlc_mcp_server
```

## Code Quality

Format code with Black:

```bash
black aws_samples tests
```

Sort imports with isort:

```bash
isort aws_samples tests
```

Run linting with Ruff:

```bash
ruff check aws_samples tests
```

Run type checking with Pyright:

```bash
pyright aws_samples
```

## Running the Server Locally

Start the server in development mode:

```bash
FASTMCP_LOG_LEVEL=DEBUG python -m aws_samples.dlc_mcp_server.main
```

Or using the installed command:

```bash
FASTMCP_LOG_LEVEL=DEBUG dlc-mcp-server
```

## Project Structure

```
sample-dlc-mcp-server/
├── aws_samples/dlc_mcp_server/
│   ├── modules/           # Feature modules
│   │   ├── best_practices.py
│   │   ├── containers.py
│   │   ├── deployment.py
│   │   ├── dlc_discovery.py
│   │   ├── image_building.py
│   │   ├── instance_advisor.py
│   │   ├── troubleshooting.py
│   │   └── upgrade.py
│   ├── templates/         # Jinja2 templates
│   ├── utils/             # Utility functions
│   │   ├── aws_utils.py
│   │   ├── config.py
│   │   ├── dlc_images.py  # Dynamic DLC catalog
│   │   ├── docker_utils.py
│   │   └── security.py
│   └── main.py            # Server entry point
├── examples/              # Usage examples
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

## Key Features

- **Dynamic DLC Catalog**: Images are fetched from [AWS DLC GitHub](https://aws.github.io/deep-learning-containers/reference/available_images/) and cached for 1 hour
- **No AWS Credentials Required**: Discovery and recommendation tools work without AWS credentials
- **MCP Protocol**: Implements Model Context Protocol for IDE integration

## Adding New Features

1. Create a new module in `aws_samples/dlc_mcp_server/modules/`
2. Implement the tool functions
3. Register the module in `main.py` using `register_module(mcp)`
4. Add tests in `tests/`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
