# RVSNUpy

**RVSNUpy** is a Python package for spectroscopic redshift measurement based on inverse-variance-weighted cross-correlation.

## 🔧 Installation

### Download

To download RVSNUpy from GitHub:

```bash
git clone https://github.com/kim8517/RVSNUpy.git
```

### Environment Variable

RVSNUpy requires an environment variable `rvsnupy` that points to the root directory of this project.

For Bash or Zsh, add the following line to your `~/.bashrc` or `~/.zshrc`:

```bash
export rvsnupy=/full/path/to/RVSNUpy  # replace with your actual path
```

Then reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

You can confirm it’s set with:

```bash
echo $rvsnupy
```

### Installation
```bash
cd RVSNUpy
pip install .
```

## How to use?

### Quick start
```python
# Read sdss spectrum
from RVSNUpy.spectrum_import import sdss_fits
sdss_spec = sdss_fits('spectra/sdss_spec1.fits')

# Read MMT/Hectospec spectrum
from RVSNUpy.spectrum_import import MMT_raw
mmt_spec = MMT_raw('spectra/mmt_spec1.fits')

# Import a template set
from RVSNUpy.template import sdss_galaxy_templates
gal_temps = sdss_galaxy_templates("vacuum")

# Initialize a rvm
from RVSNUpy.rvm import rvm
measure = rvm(gal_temps) # You need to initialize the rvm with a template set only once if you use the same template for all measurements.

# Single measurement
df=measure.z_single(sdss_spec)
print(df) # Redshift measurement based on all templates. The measurement with 'best' in the note is considered the best measurement.

# Multiple measurements
df=measure.z_speclist([sdss_spec, mmt_spec])
print(df)
```

For detailed use, refer to .ipynb files in example, espeically z_measure.ipynb.
An example for analyzing the redshift measurement result—including visually investigating the spectrum and cross-correlation signals—also will be added soon.

## 📄 License

This project is licensed under the **MIT License**.
