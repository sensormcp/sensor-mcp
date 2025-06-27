# SensorMCP Server

A **SensorMCP Model Context Protocol (MCP) Server** that enables automated dataset creation and custom object detection model training through natural language interactions. This project integrates computer vision capabilities with Large Language Models using the MCP standard.

## 🌟 About

**SensorMCP Server** combines the power of foundation models (like GroundedSAM) with custom model training (YOLOv8) to create a seamless workflow for object detection. Using the Model Context Protocol, it enables LLMs to:

- Automatically label images using foundation models
- Create custom object detection datasets
- Train specialized detection models
- Download images from Unsplash for training data

> [!NOTE] The Model Context Protocol (MCP) enables seamless integration between LLMs and external tools, making this ideal for AI-powered computer vision workflows.

## ✨ Features

- **Foundation Model Integration**: Uses GroundedSAM for automatic image labeling
- **Custom Model Training**: Fine-tune YOLOv8 models on your specific objects
- **Image Data Management**: Download images from Unsplash or import local images
- **Ontology Definition**: Define custom object classes through natural language
- **MCP Protocol**: Native integration with LLM workflows and chat interfaces
- **Fixed Data Structure**: Organized directory layout for reproducible workflows

## 🛠️ Installation

### Prerequisites

- **uv** for package management
- **Python 3.13+** (`uv python install 3.13`)
- **CUDA-compatible GPU** (recommended for training)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd sensor-mcp
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Set up environment variables** (create `.env` file):
```bash
UNSPLASH_API_KEY=your_unsplash_api_key_here
```

## 🚀 Usage

### Running the MCP Server

**For MCP integration** (recommended):
```bash
uv run src/zoo_mcp.py
```

**For standalone web server**:
```bash
uv run src/server.py
```

### MCP Configuration

Add to your MCP client configuration:
```json
{
    "mcpServers": {
        "sensormcp-server": {
            "type": "stdio",
            "command": "uv",
            "args": [
                "--directory",
                "/path/to/sensor-mcp",
                "run",
                "src/zoo_mcp.py"
            ]
        }
    }
}
```

### Available MCP Tools

1. **list_available_models()** - View supported base and target models
2. **define_ontology(objects_list)** - Define object classes to detect
3. **set_base_model(model_name)** - Initialize foundation model for labeling
4. **set_target_model(model_name)** - Initialize target model for training
5. **fetch_unsplash_images(query, max_images)** - Download training images
6. **import_images_from_folder(folder_path)** - Import local images
7. **label_images()** - Auto-label images using the base model
8. **train_model(epochs, device)** - Train custom detection model

### Example Workflow

Through your MCP-enabled LLM interface:

1. **Define what to detect**:
   ```
   Define ontology for "tiger, elephant, zebra"
   ```

2. **Set up models**:
   ```
   Set base model to grounded_sam
   Set target model to yolov8n.pt
   ```

3. **Get training data**:
   ```
   Fetch 50 images from Unsplash for "wildlife animals"
   ```

4. **Create dataset**:
   ```
   Label all images using the base model
   ```

5. **Train custom model**:
   ```
   Train model for 100 epochs on device 0
   ```

## 📁 Project Structure

```
sensor-mcp/
├── src/
│   ├── server.py          # Main MCP server implementation
│   ├── zoo_mcp.py         # MCP entry point
│   ├── models.py          # Model management and training
│   ├── image_utils.py     # Image processing and Unsplash API
│   ├── state.py           # Application state management
│   └── data/              # Created automatically
│       ├── raw_images/    # Original/unlabeled images
│       ├── labeled_images/# Auto-labeled datasets  
│       └── models/        # Trained model weights
├── static/                # Web interface assets
└── index.html            # Web interface template
```

## 🔧 Supported Models

### Base Models (for auto-labeling)
- **GroundedSAM**: Foundation model for object detection and segmentation

### Target Models (for training)
- **YOLOv8n.pt**: Nano - fastest inference
- **YOLOv8s.pt**: Small - balanced speed/accuracy
- **YOLOv8m.pt**: Medium - higher accuracy
- **YOLOv8l.pt**: Large - high accuracy
- **YOLOv8x.pt**: Extra Large - highest accuracy

## 🌐 API Integration

### Unsplash API
To use image download functionality:
1. Create an account at [Unsplash Developers](https://unsplash.com/developers)
2. Create a new application
3. Add your access key to the `.env` file

## 🛠️ Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black src/
```

## 📋 Requirements

See `pyproject.toml` for full dependency list. Key dependencies:
- `mcp[cli]` - Model Context Protocol
- `autodistill` - Foundation model integration
- `torch` & `torchvision` - Deep learning framework
- `ultralytics` - YOLOv8 implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📖 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{Guo2025,
  author = {Guo, Yunqi and Zhu, Guanyu and Liu, Kaiwei and Xing, Guoliang},
  title = {A Model Context Protocol Server for Custom Sensor Tool Creation},
  booktitle = {3rd International Workshop on Networked AI Systems (NetAISys '25)},
  year = {2025},
  month = {jun},
  address = {Anaheim, CA, USA},
  publisher = {ACM},
  doi = {10.1145/3711875.3736687},
  isbn = {979-8-4007-1453-5/25/06}
}
```

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions about the zoo dataset mentioned in development:
**Email**: yq@anysign.net