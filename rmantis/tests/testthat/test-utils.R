test_that("rmantis_check_python() stops when Python exe does not exist", {
  mockery::stub(rmantis_check_python, "reticulate::py_config",
                function() list(python = "/nonexistent/python"))

  expect_error(
    rmantis_check_python(),
    regexp = "Python executable not found"
  )
})

test_that("rmantis_check_python() stops when mantis module is missing", {
  mockery::stub(rmantis_check_python, "reticulate::py_config",
                function() list(python = R.home("bin/Rscript")))
  mockery::stub(rmantis_check_python, "reticulate::py_module_available",
                function(m) FALSE)

  expect_error(
    rmantis_check_python(),
    regexp = "mantis.*module is not installed"
  )
})

test_that("rmantis_check_python() returns cfg invisibly on success", {
  fake_cfg <- list(python = R.home("bin/Rscript"))
  mockery::stub(rmantis_check_python, "reticulate::py_config",
                function() fake_cfg)
  mockery::stub(rmantis_check_python, "reticulate::py_module_available",
                function(m) TRUE)

  result <- rmantis_check_python()
  expect_identical(result, fake_cfg)
})
