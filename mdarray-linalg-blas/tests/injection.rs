//! Tests for the injection feature
//!
//! Note: Full end-to-end testing with actual injected function pointers
//! requires a specific setup (e.g., loading OpenBLAS dynamically).
//! These tests verify the API surface and registration logic.

#![cfg(feature = "injection")]

use std::collections::HashMap;

use mdarray_linalg_blas::{
    RegistrationError, SUPPORTED_FUNCTIONS, available_functions, is_backend_registered,
};

// Note: We cannot easily test injected_gemm without actual BLAS function pointers.
// Real integration tests would be done in the consuming crate (e.g., tensor4all-rs)
// where we can inject pointers from scipy or libblastrampoline.

#[test]
fn test_api_surface() {
    // Just verify the types and functions exist and compile
    let _: bool = is_backend_registered();

    // Check available_functions returns None when no backend registered
    // (may fail if another test registered a backend first due to test parallelism)
    if !is_backend_registered() {
        assert!(available_functions().is_none());
    }
}

#[test]
fn test_supported_functions() {
    // Verify the supported functions list
    assert!(SUPPORTED_FUNCTIONS.contains(&"dgemm_"));
    assert!(SUPPORTED_FUNCTIONS.contains(&"zgemm_"));
    assert!(SUPPORTED_FUNCTIONS.contains(&"sgemm_"));
    assert!(SUPPORTED_FUNCTIONS.contains(&"cgemm_"));
}

#[test]
fn test_registration_error() {
    let err = RegistrationError::AlreadyRegistered;
    assert_eq!(format!("{}", err), "BLAS backend already registered");

    let err = RegistrationError::UnknownFunction("unknown_".to_string());
    assert!(format!("{}", err).contains("unknown_"));
}

#[test]
fn test_hashmap_api() {
    // Test that HashMap API compiles correctly
    let fns: HashMap<&str, *const ()> = HashMap::new();
    // Can't actually register without valid pointers, but verify type signature
    let _ = fns;
}
