# Medhok – Javanese Dialect Identification Project

This repository contains code and resources for the Javanese Dialect Identification project. The project aims to identify dialects from audio data using various machine learning and deep learning models.

## Project Structure

- `medhok/` - Main package containing core code.
- `exploratory/` - Experimental scripts and notebooks.
- `docs/` - Documentation and performance reports.
- `tests/` - Unit and integration tests.
- `visualization/` - Scripts and resources for visualizing results.
- `model_backup/` - Backup of trained models.
- `log/` - Log files and TensorBoard logs.
- `notes/` - Project notes and references.

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Configuration

Project-wide constants and paths are defined in [`medhok/configs/config.py`](medhok/configs/config.py).

## Running the Project

1. Prepare your dataset in the `data/` directory.
2. Configure parameters in the config files as needed.
3. Run training or evaluation scripts from the `medhok/` or `exploratory/` directories.

## Documentation

See the [docs/](docs/) directory for model performance reports and visualizations.

## Project Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure project paths:**
   - All main paths are set in [`medhok/configs/config.py`](medhok/configs/config.py).
   - By default, the project expects a `data/` directory at the root, with audio files in `data/audio/`.

---

## Dataset Setup

This project uses the Javanese Dialect Mapping Project dataset from the Max Planck Institute for Evolutionary Anthropology.

- **About the dataset:**  
  [Javanese Dialect Mapping Project Info](https://www.eva.mpg.de/linguistics/past-research-resources/documentation-and-description/the-javanese-dialect-mapping-project/?Fsize=1&cHash=c4453d05f7b8ce4f2bf3ee41819c3b15)

- **Download link:**  
  [MPI Archive Download](https://archive.mpi.nl/tla/islandora/object/tla%3A1839_00_0000_0000_0022_75C3_1)

**Important:**  
The audio files are very large (over 50GB). You must download the dataset manually from the MPI website and follow their instructions for access and usage.

**After downloading:**
1. Extract the audio files.
2. Place them in the following directory structure at the project root:
   ```
   data/
     audio/
       <dialect1>/
         *.wav
       <dialect2>/
         *.wav
       ...
   ```
   - Each subdirectory under `audio/` should correspond to a dialect or region as organized in the dataset.
   - Ensure the `data/audio/` path matches the configuration in `medhok/configs/config.py`.

For more details on the dataset, refer to the official [MPI documentation](https://www.eva.mpg.de/linguistics/past-research-resources/documentation-and-description/the-javanese-dialect-mapping-project/?Fsize=1&cHash=c4453d05f7b8ce4f2bf3ee41819c3b15).

## License

See `LICENSE` file (if present) for license information.