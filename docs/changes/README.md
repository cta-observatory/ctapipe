# How to use towncrier

An overview can be found [here](https://towncrier.readthedocs.io/en/stable/quickstart.html#).

1. Create a new file for your changes `<PULL REQUEST>.<TYPE>.rst` in the corresponding folder. The following types are available:
    - feature: `New feature`
    - bugfix: `Bugfix`
    - api: `API Changes`
    - datamodel: `Data Model Changes`
    - optimization: `Refactoring and Optimization`
    - maintenance: `Maintenance`


2. Write a suitable message for the change:
    ```
    Fixed ``crazy_function`` to be consistent with ``not_so_crazy_function``
    ```

3. (For maintainers) How to generate a change log:
    - Execute the following command in the base directory of the project
    ```
    towncrier build --version=<VERSION NUMBER>
    ```
