# Helper: stubs out all external calls so mantis_forecast() can run
# without Python or a real model directory.
stub_forecast_externals <- function(dest_dir = tempdir()) {
  # Python check passes
  mockery::stub(mantis_forecast, "rmantis_check_python",
                function() list(python = R.home("bin/Rscript")))

  # Fake mantis Python module
  fake_model <- list(
    predict = function(...) matrix(1:8, nrow = 2)
  )
  fake_mantis <- list(
    Mantis = function(...) fake_model
  )
  mockery::stub(mantis_forecast, "reticulate::import",
                function(m) fake_mantis)
}

# ── Input validation ──────────────────────────────────────────────────────────

test_that("mantis_forecast() errors on empty time_series", {
  expect_error(
    mantis_forecast(numeric(0), model_dir = tempdir()),
    regexp = "non-empty numeric vector"
  )
})

test_that("mantis_forecast() errors on non-numeric time_series", {
  expect_error(
    mantis_forecast(c("a", "b"), model_dir = tempdir()),
    regexp = "non-empty numeric vector"
  )
})

test_that("mantis_forecast() errors on invalid horizon", {
  expect_error(
    mantis_forecast(1:10, model_dir = tempdir(), horizon = 5L),
    regexp = "horizon.*must be either 4 or 8"
  )
})

test_that("mantis_forecast() errors when model_dir does not exist", {
  mockery::stub(mantis_forecast, "rmantis_check_python",
                function() list(python = R.home("bin/Rscript")))

  expect_error(
    mantis_forecast(1:10, model_dir = "/nonexistent/path"),
    regexp = "Model directory not found"
  )
})

# ── Successful forecast ───────────────────────────────────────────────────────

test_that("mantis_forecast() returns a matrix on valid inputs", {
  stub_forecast_externals()

  result <- mantis_forecast(1:20, model_dir = tempdir(), horizon = 4L)
  expect_true(is.matrix(result))
})

test_that("mantis_forecast() passes covariate = NULL when not supplied", {
  capture <- list()
  mockery::stub(mantis_forecast, "rmantis_check_python",
                function() list(python = R.home("bin/Rscript")))

  fake_model <- list(
    predict = function(...) {
      capture <<- list(...)
      matrix(1:4, nrow = 2)
    }
  )
  mockery::stub(mantis_forecast, "reticulate::import",
                function(m) list(Mantis = function(...) fake_model))

  mantis_forecast(1:10, model_dir = tempdir(), covariate = NULL)
  expect_null(capture$covariate)
})

test_that("mantis_forecast() debug mode prints messages without error", {
  stub_forecast_externals()

  expect_output(
    mantis_forecast(1:10, model_dir = tempdir(), debug = TRUE),
    regexp = "Python executable"
  )
})
