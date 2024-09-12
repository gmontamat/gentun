# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Here are areas that can be improved in this library:

- Add [genes](src/gentun/genes.py#L11-L47) and [models](src/gentun/models/base.py#L9-L25) to
support more paper implementation
- Add a method to share the training data between the controller node and workers
- Add some type of proof-of-work validation for workers

You can also help us speed up hyperparameter search by contributing your spare GPU time.

There was a major refactor of this library in 2024, the old version is still available in
the [`old` branch](https://github.com/gmontamat/gentun/tree/old). Some cool forks added
features to this release.

## Pull Request Process

1. Ensure your branch is linted with `black` and `isort` using
   the [pyproject.toml](./pyproject.toml) configurations.
2. Update the README.md with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. Use [The Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/) in
   your PR title and description.