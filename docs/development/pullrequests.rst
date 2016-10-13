**********************************
Making and Accepting Pull Requests
**********************************

Making a Pull Request
=====================

In the pull request description (editable on GitHub), you should
include the following:

* Description of the new features, changes, or refactorings
  included in the pull request
* If there is a new feature, include a brief *use case* for the feature
* give an example of use if applicable


Accepting Pull Requests
=======================

`ctapipe` maintainers must do a code review before accepting any pull
request. The following guidelines should be used to facilitate the
procedure:

* Perform a Scientific or Conceptual Review
 * look at the use case for the proposed change. Is it clear? If not, ask for improvement
 * 
* Perform a Code Review
 * Check that all functions and classes have API documentation in the
   correct format (and check that there are no warnings generated when
   building the docs)
 * Check that there are at least basic unit tests for each function and class
 * Run all unit tests.
  * If any unit tests included in the pull request fail, ask the develper if that is normal (sometimes there are tests intended to fail until a feature is available or implemented)
  * If any other unit tests that previously suceeded now fail, **stop** and ask the develper to find out what was broken by their change
 * Read through the new code being contributed, and see that it follows the style guidelines and that the API
 * Check that the API (function and class definitions) is clear and easy to understand. If not, ask for it to be cleaned up
 * Check that the code uses the existing features of the ctapipe framework  
 * Check that the code doesn't introduce new features that are already present in another form in the framework

