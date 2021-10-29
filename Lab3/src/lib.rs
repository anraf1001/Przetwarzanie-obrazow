use ndarray::{s, Array2, ArrayD, ArrayView2, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn kuwahara_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn apply_kuwahara(image: ArrayViewD<'_, u8>, window_size: usize) -> ArrayD<u8> {
        let border_size = window_size / 2;
        let image_border = create_image_with_padding(&image, border_size);
        let mut image_new = ArrayD::zeros(image.shape());

        for y in 0..image_new.shape()[0] {
            for x in 0..image_new.shape()[1] {
                let window = image_border.slice(s![y..y + window_size, x..x + window_size]);
                let regions = [
                    window.slice(s![0..border_size + 1, 0..border_size + 1]),
                    window.slice(s![border_size..window_size, 0..border_size + 1]),
                    window.slice(s![0..border_size + 1, border_size..window_size]),
                    window.slice(s![border_size..window_size, border_size..window_size]),
                ];

                let mut regions_std_mean = [
                    std_mean(regions[0]),
                    std_mean(regions[1]),
                    std_mean(regions[2]),
                    std_mean(regions[3]),
                ];

                regions_std_mean.sort_by(|lhs, rhs| lhs.0.partial_cmp(&rhs.0).unwrap());

                image_new[[y, x]] = regions_std_mean[0].1 as u8;
            }
        }

        image_new
    }

    #[pyfn(m)]
    #[pyo3(name = "apply_kuwahara")]
    fn apply_kuwahara_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArrayDyn<u8>,
        window_size: usize,
    ) -> &'py PyArrayDyn<u8> {
        let image = image.as_array();
        apply_kuwahara(image, window_size).into_pyarray(py)
    }

    Ok(())
}

fn mean(region: ArrayView2<u8>) -> f64 {
    let sum: f64 = region.iter().map(|x| *x as f64).sum();
    let count = region.len();

    sum / count as f64
}

fn std_mean(region: ArrayView2<u8>) -> (f64, f64) {
    let region_mean = mean(region);
    let count = region.len();

    let variance = region
        .iter()
        .map(|value| {
            let diff = region_mean - (*value as f64);

            diff * diff
        })
        .sum::<f64>()
        / count as f64;

    (variance.sqrt(), region_mean)
}

fn create_image_with_padding(image: &ArrayViewD<u8>, border_size: usize) -> Array2<u8> {
    let mut image_border = Array2::<u8>::zeros([
        image.shape()[0] + 2 * border_size,
        image.shape()[1] + 2 * border_size,
    ]);

    let mut ori_region = image_border.slice_mut(s![
        border_size..image.shape()[0] - border_size,
        border_size..image.shape()[1] - border_size
    ]);

    for y in 0..ori_region.shape()[0] {
        for x in 0..ori_region.shape()[1] {
            ori_region[[y, x]] = image[[y, x]];
        }
    }

    image_border
}
