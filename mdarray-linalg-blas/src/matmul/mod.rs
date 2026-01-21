pub mod context;
pub mod scalar;
pub mod simple;

#[cfg(feature = "injection")]
pub mod injection;

pub use simple::{gemm, gemm_uninit};

#[cfg(feature = "injection")]
pub use injection::{
    AvailableFunctions, InjectedBackend, InjectedBlas, InjectedBlasScalar, RegistrationError,
    SUPPORTED_FUNCTIONS, available_functions, get_backend, injected_gemm, injected_gemm_uninit,
    is_backend_registered, register_backend,
};
