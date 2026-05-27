test_that("mantis_download_weights() errors on invalid horizon", {
  expect_error(
    mantis_download_weights(horizon = 3L),
    regexp = "horizon must be either 4 or 8"
  )
  expect_error(
    mantis_download_weights(horizon = 16L),
    regexp = "horizon must be either 4 or 8"
  )
})

test_that("mantis_download_weights() constructs correct filename for cov/nocov", {
  mockery::stub(mantis_download_weights, "utils::download.file",
                function(...) invisible(NULL))
  mockery::stub(mantis_download_weights, "message",
                function(...) invisible(NULL))

  td <- tempdir()

  result_cov <- mantis_download_weights(
    horizon = 4L, use_covariate = TRUE, dest_dir = td
  )
  expect_match(result_cov, "mantis_4w_cov\\.pt$")

  result_nocov <- mantis_download_weights(
    horizon = 8L, use_covariate = FALSE, dest_dir = td
  )
  expect_match(result_nocov, "mantis_8w_nocov\\.pt$")
})

test_that("mantis_download_weights() creates dest_dir if missing", {
  mockery::stub(mantis_download_weights, "utils::download.file",
                function(...) invisible(NULL))
  mockery::stub(mantis_download_weights, "message",
                function(...) invisible(NULL))

  td <- file.path(tempdir(), "rmantis_test_dir", format(Sys.time(), "%s"))
  on.exit(unlink(td, recursive = TRUE), add = TRUE)

  mantis_download_weights(horizon = 4L, dest_dir = td)
  expect_true(dir.exists(td))
})

test_that("mantis_download_weights() returns dest path invisibly", {
  mockery::stub(mantis_download_weights, "utils::download.file",
                function(...) invisible(NULL))
  mockery::stub(mantis_download_weights, "message",
                function(...) invisible(NULL))

  td     <- tempdir()
  result <- withVisible(mantis_download_weights(horizon = 4L, dest_dir = td))
  expect_false(result$visible)
  expect_type(result$value, "character")
})
