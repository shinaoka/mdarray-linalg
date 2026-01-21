//! Runtime injection of BLAS function pointers for Python/Julia integration.
//!
//! This module provides the ability to inject BLAS function pointers at runtime,
//! enabling seamless integration with host language BLAS implementations:
//! - Python: via scipy.linalg.cython_blas
//! - Julia: via libblastrampoline (lbt_get_forward)
//!
//! # Example
//!
//! ```ignore
//! use std::collections::HashMap;
//! use mdarray_linalg_blas::matmul::injection::register_backend;
//!
//! // Get function pointers from host language (e.g., via dlsym or PyCapsule)
//! let mut fns = HashMap::new();
//! fns.insert("dgemm_", dgemm_ptr as *const ());
//! fns.insert("zgemm_", zgemm_ptr as *const ());
//!
//! // Register all functions at once (LP64 = 32-bit integers)
//! register_backend(fns, false).expect("Backend already registered");
//!
//! // Now use injected_gemm for computations
//! injected_gemm(1.0, &a, &b, 0.0, &mut c);
//! ```

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::OnceLock;

use mdarray::{Dim, Layout, Slice, Tensor};
use mdarray_linalg::{dims3, into_i32, trans_stride};
use num_complex::{Complex, ComplexFloat};

// ============================================================================
// Function pointer type definitions (Fortran BLAS interface)
// ============================================================================

// LP64 types (32-bit integers) - used by Python/scipy
pub type DgemmLp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const f64,
    a: *const f64,
    lda: *const i32,
    b: *const f64,
    ldb: *const i32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const i32,
);

pub type ZgemmLp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const Complex<f64>,
    a: *const Complex<f64>,
    lda: *const i32,
    b: *const Complex<f64>,
    ldb: *const i32,
    beta: *const Complex<f64>,
    c: *mut Complex<f64>,
    ldc: *const i32,
);

pub type SgemmLp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const f32,
    a: *const f32,
    lda: *const i32,
    b: *const f32,
    ldb: *const i32,
    beta: *const f32,
    c: *mut f32,
    ldc: *const i32,
);

pub type CgemmLp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const Complex<f32>,
    a: *const Complex<f32>,
    lda: *const i32,
    b: *const Complex<f32>,
    ldb: *const i32,
    beta: *const Complex<f32>,
    c: *mut Complex<f32>,
    ldc: *const i32,
);

// ILP64 types (64-bit integers) - used by Julia with USE_BLAS64=true
pub type DgemmIlp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const f64,
    a: *const f64,
    lda: *const i64,
    b: *const f64,
    ldb: *const i64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const i64,
);

pub type ZgemmIlp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const Complex<f64>,
    a: *const Complex<f64>,
    lda: *const i64,
    b: *const Complex<f64>,
    ldb: *const i64,
    beta: *const Complex<f64>,
    c: *mut Complex<f64>,
    ldc: *const i64,
);

pub type SgemmIlp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const f32,
    a: *const f32,
    lda: *const i64,
    b: *const f32,
    ldb: *const i64,
    beta: *const f32,
    c: *mut f32,
    ldc: *const i64,
);

pub type CgemmIlp64 = unsafe extern "C" fn(
    transa: *const i8,
    transb: *const i8,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const Complex<f32>,
    a: *const Complex<f32>,
    lda: *const i64,
    b: *const Complex<f32>,
    ldb: *const i64,
    beta: *const Complex<f32>,
    c: *mut Complex<f32>,
    ldc: *const i64,
);

// ============================================================================
// Backend storage
// ============================================================================

/// Finalized backend with all registered function pointers.
/// Stored in OnceLock for ~0.5ns access overhead.
#[derive(Default)]
pub struct InjectedBackend {
    // LP64 functions
    pub dgemm_lp64: Option<DgemmLp64>,
    pub zgemm_lp64: Option<ZgemmLp64>,
    pub sgemm_lp64: Option<SgemmLp64>,
    pub cgemm_lp64: Option<CgemmLp64>,

    // ILP64 functions
    pub dgemm_ilp64: Option<DgemmIlp64>,
    pub zgemm_ilp64: Option<ZgemmIlp64>,
    pub sgemm_ilp64: Option<SgemmIlp64>,
    pub cgemm_ilp64: Option<CgemmIlp64>,

    /// True if this backend uses ILP64 (64-bit integers)
    pub is_ilp64: bool,
}

/// Global backend storage using OnceLock for minimal overhead (~0.5ns per access)
static BACKEND: OnceLock<InjectedBackend> = OnceLock::new();

// ============================================================================
// Registration API
// ============================================================================

/// Error type for backend registration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationError {
    /// Backend was already registered
    AlreadyRegistered,
    /// Unknown function name
    UnknownFunction(String),
}

impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRegistered => write!(f, "BLAS backend already registered"),
            Self::UnknownFunction(name) => write!(f, "Unknown BLAS function: {}", name),
        }
    }
}

impl std::error::Error for RegistrationError {}

/// Register BLAS function pointers from a HashMap.
///
/// # Arguments
/// * `functions` - Map of function names (e.g., "dgemm_") to function pointers
/// * `is_ilp64` - True if using ILP64 (64-bit integers), false for LP64 (32-bit)
///
/// # Function names
/// Standard Fortran BLAS names with trailing underscore:
/// - `dgemm_`, `zgemm_`, `sgemm_`, `cgemm_` (matrix-matrix multiply)
///
/// # Returns
/// `Ok(())` on success, `Err(RegistrationError)` if already registered or unknown function
///
/// # Example
/// ```ignore
/// let mut fns = HashMap::new();
/// fns.insert("dgemm_", dgemm_ptr as *const ());
/// fns.insert("zgemm_", zgemm_ptr as *const ());
/// register_backend(fns, false)?;
/// ```
#[allow(clippy::missing_transmute_annotations)]
pub fn register_backend(
    functions: HashMap<&str, *const ()>,
    is_ilp64: bool,
) -> Result<(), RegistrationError> {
    let mut backend = InjectedBackend {
        is_ilp64,
        ..Default::default()
    };

    for (name, ptr) in functions {
        if is_ilp64 {
            match name {
                "dgemm_" => backend.dgemm_ilp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "zgemm_" => backend.zgemm_ilp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "sgemm_" => backend.sgemm_ilp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "cgemm_" => backend.cgemm_ilp64 = Some(unsafe { std::mem::transmute(ptr) }),
                _ => return Err(RegistrationError::UnknownFunction(name.to_string())),
            }
        } else {
            match name {
                "dgemm_" => backend.dgemm_lp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "zgemm_" => backend.zgemm_lp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "sgemm_" => backend.sgemm_lp64 = Some(unsafe { std::mem::transmute(ptr) }),
                "cgemm_" => backend.cgemm_lp64 = Some(unsafe { std::mem::transmute(ptr) }),
                _ => return Err(RegistrationError::UnknownFunction(name.to_string())),
            }
        }
    }

    BACKEND
        .set(backend)
        .map_err(|_| RegistrationError::AlreadyRegistered)
}

/// Check if a backend is registered
pub fn is_backend_registered() -> bool {
    BACKEND.get().is_some()
}

/// Get the registered backend, if any
pub fn get_backend() -> Option<&'static InjectedBackend> {
    BACKEND.get()
}

/// Query which functions are available in the registered backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct AvailableFunctions {
    pub dgemm: bool,
    pub zgemm: bool,
    pub sgemm: bool,
    pub cgemm: bool,
    pub is_ilp64: bool,
}

/// Get status of available BLAS functions.
/// Returns None if no backend is registered.
pub fn available_functions() -> Option<AvailableFunctions> {
    BACKEND.get().map(|b| {
        if b.is_ilp64 {
            AvailableFunctions {
                dgemm: b.dgemm_ilp64.is_some(),
                zgemm: b.zgemm_ilp64.is_some(),
                sgemm: b.sgemm_ilp64.is_some(),
                cgemm: b.cgemm_ilp64.is_some(),
                is_ilp64: true,
            }
        } else {
            AvailableFunctions {
                dgemm: b.dgemm_lp64.is_some(),
                zgemm: b.zgemm_lp64.is_some(),
                sgemm: b.sgemm_lp64.is_some(),
                cgemm: b.cgemm_lp64.is_some(),
                is_ilp64: false,
            }
        }
    })
}

/// List of supported function names for registration
pub const SUPPORTED_FUNCTIONS: &[&str] = &["dgemm_", "zgemm_", "sgemm_", "cgemm_"];

// ============================================================================
// Helper functions
// ============================================================================

/// Marker struct for the injected BLAS backend.
#[derive(Default, Clone, Copy)]
pub struct InjectedBlas;

// Helper to convert CBLAS transpose to Fortran BLAS character
fn cblas_trans_to_char(trans: cblas_sys::CBLAS_TRANSPOSE) -> i8 {
    match trans {
        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans => b'N' as i8,
        cblas_sys::CBLAS_TRANSPOSE::CblasTrans => b'T' as i8,
        cblas_sys::CBLAS_TRANSPOSE::CblasConjTrans => b'C' as i8,
    }
}

// Helper to check if layout is row-major
fn is_row_major(layout: cblas_sys::CBLAS_LAYOUT) -> bool {
    match layout {
        cblas_sys::CBLAS_LAYOUT::CblasRowMajor => true,
        cblas_sys::CBLAS_LAYOUT::CblasColMajor => false,
    }
}

// ============================================================================
// InjectedBlasScalar trait and implementations
// ============================================================================

/// Trait for types that can use the injected BLAS backend
pub trait InjectedBlasScalar: Sized {
    /// Call GEMM using the injected backend
    ///
    /// # Safety
    /// Caller must ensure all pointers are valid and dimensions are correct.
    ///
    /// # Panics
    /// Panics if no backend is registered or if the required function is not available.
    #[allow(clippy::too_many_arguments)]
    unsafe fn injected_gemm(
        layout: cblas_sys::CBLAS_LAYOUT,
        transa: cblas_sys::CBLAS_TRANSPOSE,
        transb: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
}

macro_rules! impl_injected_blas_scalar {
    ($ty:ty, $lp64_field:ident, $ilp64_field:ident, $name:literal) => {
        impl InjectedBlasScalar for $ty {
            unsafe fn injected_gemm(
                layout: cblas_sys::CBLAS_LAYOUT,
                transa: cblas_sys::CBLAS_TRANSPOSE,
                transb: cblas_sys::CBLAS_TRANSPOSE,
                m: i32,
                n: i32,
                k: i32,
                alpha: Self,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                beta: Self,
                c: *mut Self,
                ldc: i32,
            ) {
                let backend = BACKEND
                    .get()
                    .expect("BLAS backend not registered. Call register_backend() first.");

                // Convert from CBLAS (row/column major + trans) to Fortran BLAS interface
                // Fortran BLAS is always column-major, so we need to swap for row-major
                let (
                    actual_transa,
                    actual_transb,
                    actual_m,
                    actual_n,
                    actual_a,
                    actual_b,
                    actual_lda,
                    actual_ldb,
                ) = if is_row_major(layout) {
                    // For row-major: C = A*B becomes C^T = B^T * A^T
                    (
                        cblas_trans_to_char(transb),
                        cblas_trans_to_char(transa),
                        n,
                        m,
                        b,
                        a,
                        ldb,
                        lda,
                    )
                } else {
                    (
                        cblas_trans_to_char(transa),
                        cblas_trans_to_char(transb),
                        m,
                        n,
                        a,
                        b,
                        lda,
                        ldb,
                    )
                };

                if backend.is_ilp64 {
                    let func = backend.$ilp64_field.expect(concat!(
                        $name,
                        " (ILP64) not available in registered backend"
                    ));
                    let m64 = actual_m as i64;
                    let n64 = actual_n as i64;
                    let k64 = k as i64;
                    let lda64 = actual_lda as i64;
                    let ldb64 = actual_ldb as i64;
                    let ldc64 = ldc as i64;
                    unsafe {
                        func(
                            &actual_transa,
                            &actual_transb,
                            &m64,
                            &n64,
                            &k64,
                            &alpha as *const Self,
                            actual_a,
                            &lda64,
                            actual_b,
                            &ldb64,
                            &beta as *const Self,
                            c,
                            &ldc64,
                        );
                    }
                } else {
                    let func = backend.$lp64_field.expect(concat!(
                        $name,
                        " (LP64) not available in registered backend"
                    ));
                    unsafe {
                        func(
                            &actual_transa,
                            &actual_transb,
                            &actual_m,
                            &actual_n,
                            &k,
                            &alpha as *const Self,
                            actual_a,
                            &actual_lda,
                            actual_b,
                            &actual_ldb,
                            &beta as *const Self,
                            c,
                            &ldc,
                        );
                    }
                }
            }
        }
    };
}

impl_injected_blas_scalar!(f64, dgemm_lp64, dgemm_ilp64, "dgemm_");
impl_injected_blas_scalar!(f32, sgemm_lp64, sgemm_ilp64, "sgemm_");
impl_injected_blas_scalar!(Complex<f64>, zgemm_lp64, zgemm_ilp64, "zgemm_");
impl_injected_blas_scalar!(Complex<f32>, cgemm_lp64, cgemm_ilp64, "cgemm_");

// ============================================================================
// High-level GEMM functions
// ============================================================================

/// GEMM using injected BLAS backend.
///
/// This function reuses the same stride/layout logic as `simple::gemm`
/// but delegates to the injected function pointers.
///
/// # Panics
/// Panics if no backend is registered or if the required function is not available.
pub fn injected_gemm<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    c: &mut Slice<T, (D0, D2), Lc>,
) where
    T: InjectedBlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, k) = dims3(*a.shape(), *b.shape(), *c.shape());

    let row_major = c.stride(1) == 1;
    assert!(
        row_major || c.stride(0) == 1,
        "c must be contiguous in one dimension"
    );

    let (same_order, other_order) = if row_major {
        (
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
        )
    } else {
        (
            cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
        )
    };
    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);
    let (b_trans, b_stride) = trans_stride!(b, same_order, other_order);

    let c_stride = into_i32(c.stride(if row_major { 0 } else { 1 }));

    unsafe {
        T::injected_gemm(
            if row_major {
                cblas_sys::CBLAS_LAYOUT::CblasRowMajor
            } else {
                cblas_sys::CBLAS_LAYOUT::CblasColMajor
            },
            a_trans,
            b_trans,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            beta,
            c.as_mut_ptr(),
            c_stride,
        )
    }
}

/// GEMM with uninitialized output using injected BLAS backend.
///
/// # Panics
/// Panics if no backend is registered or if the required function is not available.
pub fn injected_gemm_uninit<T, La, Lb, Lc, D0, D1, D2>(
    alpha: T,
    a: &Slice<T, (D0, D1), La>,
    b: &Slice<T, (D1, D2), Lb>,
    beta: T,
    mut c: Tensor<MaybeUninit<T>, (D0, D2)>,
) -> Tensor<T, (D0, D2)>
where
    T: InjectedBlasScalar + ComplexFloat,
    La: Layout,
    Lb: Layout,
    Lc: Layout,
    D0: Dim,
    D1: Dim,
    D2: Dim,
{
    let (m, n, k) = dims3(*a.shape(), *b.shape(), *c.shape());

    debug_assert!(c.stride(1) == 1);

    let same_order = cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans;
    let other_order = cblas_sys::CBLAS_TRANSPOSE::CblasTrans;

    let (a_trans, a_stride) = trans_stride!(a, same_order, other_order);
    let (b_trans, b_stride) = trans_stride!(b, same_order, other_order);

    let c_stride = into_i32(c.stride(0));

    unsafe {
        T::injected_gemm(
            cblas_sys::CBLAS_LAYOUT::CblasRowMajor,
            a_trans,
            b_trans,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            beta,
            c.as_mut_ptr() as *mut T,
            c_stride,
        );

        c.assume_init()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_functions_list() {
        assert!(SUPPORTED_FUNCTIONS.contains(&"dgemm_"));
        assert!(SUPPORTED_FUNCTIONS.contains(&"zgemm_"));
        assert!(SUPPORTED_FUNCTIONS.contains(&"sgemm_"));
        assert!(SUPPORTED_FUNCTIONS.contains(&"cgemm_"));
    }

    #[test]
    fn test_available_functions_struct() {
        let af = AvailableFunctions {
            dgemm: true,
            zgemm: true,
            sgemm: false,
            cgemm: false,
            is_ilp64: false,
        };
        assert!(af.dgemm);
        assert!(af.zgemm);
        assert!(!af.sgemm);
        assert!(!af.is_ilp64);
    }

    #[test]
    fn test_registration_error_display() {
        let err = RegistrationError::AlreadyRegistered;
        assert_eq!(format!("{}", err), "BLAS backend already registered");

        let err = RegistrationError::UnknownFunction("foo_".to_string());
        assert_eq!(format!("{}", err), "Unknown BLAS function: foo_");
    }

    #[test]
    fn test_injected_blas_marker() {
        let _blas = InjectedBlas::default();
        let blas1 = InjectedBlas;
        let _blas2 = blas1; // Copy
    }
}
