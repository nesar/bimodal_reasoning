import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler


def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        specs = f['raw']['spec'][()]
        z = f['raw']['z'][()]
        age = f['raw']['age'][()]
        metallicity = f['raw']['metallicity'][()]
        smass = f['raw']['smass'][()]
        wavelength = f['raw']['wavelength'][()]
    return specs, z, age, metallicity, smass, wavelength


def rescale_data(data):
    scaler = MinMaxScaler()
    rescaled = scaler.fit_transform(data)
    return rescaled, scaler


def read_all(fileIn="Data/sdss_galaxy_spec.hdf5"):
    specs, redshift, age, metallicity, smass, wavelength = load_hdf5_data(fileIn)

    print(specs.shape, redshift.shape, age.shape, metallicity.shape, smass.shape, wavelength.shape)

    # Selection: exclude bad datapoints
    selection_condition = np.where(
        (metallicity > 0) & (age > 0) & (smass > 0) & (smass < 1.e12) & (smass > 1.e8)
    )
    specs = specs[selection_condition]
    age = age[selection_condition]
    metallicity = metallicity[selection_condition]
    smass = smass[selection_condition]
    redshift = redshift[selection_condition]

    print(specs.shape, redshift.shape, age.shape, metallicity.shape, smass.shape, wavelength.shape)
    print(specs.max(), redshift.max(), age.max(), metallicity.max(), smass.max())
    print(specs.min(), redshift.min(), age.min(), metallicity.min(), smass.min())

    # Unit transforms
    age = age / 1.e9
    smass = np.log10(smass)

    print(specs.max(), redshift.max(), age.max(), metallicity.max(), smass.max())
    print(specs.min(), redshift.min(), age.min(), metallicity.min(), smass.min())

    # Min-max rescaling
    specs, scaler_specs = rescale_data(specs)
    redshift, scaler_z = rescale_data(redshift.reshape(-1, 1))
    age, scaler_age = rescale_data(age.reshape(-1, 1))
    metallicity, scaler_metallicity = rescale_data(metallicity.reshape(-1, 1))
    smass, scaler_smass = rescale_data(smass.reshape(-1, 1))

    redshift = redshift.flatten()
    age = age.flatten()
    metallicity = metallicity.flatten()
    smass = smass.flatten()

    print(specs.shape, redshift.shape, age.shape, metallicity.shape, smass.shape, wavelength.shape)
    print(specs.max(), redshift.max(), age.max(), metallicity.max(), smass.max())
    print(specs.min(), redshift.min(), age.min(), metallicity.min(), smass.min())

    X = specs
    y = np.vstack((redshift, age, metallicity, smass)).T

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    print(specs.shape, X_test.shape)
    print(specs.max(), redshift.max(), age.max(), metallicity.max(), smass.max())
    print(specs.min(), redshift.min(), age.min(), metallicity.min(), smass.min())

    return X_train, X_test, y_train, y_test, wavelength


def read_with_physical(fileIn="Data/sdss_galaxy_spec.hdf5"):
    """
    Same as read_all() but also returns un-normalized physical properties,
    aligned with the same train/test split — needed for structured verbalization.

    Returns:
        X_train, X_test       : normalized spectra (torch tensors)
        y_train_norm           : normalized targets  (torch tensor, 4 columns)
        y_phys_train           : physical properties (np.ndarray, N x 4)
                                 columns: [z, age_gyr, metallicity_Z, log_mass]
        wavelength
    """
    specs, redshift, age, metallicity, smass, wavelength = load_hdf5_data(fileIn)

    sel = np.where(
        (metallicity > 0) & (age > 0) & (smass > 0) & (smass < 1.e12) & (smass > 1.e8)
    )
    specs        = specs[sel]
    redshift     = redshift[sel]
    age          = age[sel]
    metallicity  = metallicity[sel]
    smass        = smass[sel]

    # Physical (human-readable) units — used for verbalization
    age_gyr      = age / 1.e9
    log_mass     = np.log10(smass)
    y_phys       = np.column_stack((redshift, age_gyr, metallicity, log_mass))

    # Normalized (model input)
    specs_norm, _       = rescale_data(specs)
    z_norm, _           = rescale_data(redshift.reshape(-1, 1))
    age_norm, _         = rescale_data(age_gyr.reshape(-1, 1))
    met_norm, _         = rescale_data(metallicity.reshape(-1, 1))
    mass_norm, _        = rescale_data(log_mass.reshape(-1, 1))

    y_norm = np.column_stack((z_norm, age_norm, met_norm, mass_norm))

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(specs_norm))
    idx_train, idx_test = train_test_split(idx, test_size=0.7, random_state=42)

    X_train = torch.tensor(specs_norm[idx_train], dtype=torch.float32)
    X_test  = torch.tensor(specs_norm[idx_test],  dtype=torch.float32)
    y_train_norm  = torch.tensor(y_norm[idx_train], dtype=torch.float32)
    y_phys_train  = y_phys[idx_train]            # physical, for verbalization

    return X_train, X_test, y_train_norm, y_phys_train, wavelength
