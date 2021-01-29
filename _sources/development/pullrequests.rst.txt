.. _pullrequests:

Making and Accepting Pull Requests
==================================

Making a Pull Request
---------------------

In the pull request description (editable on GitHub), you should
include the following information (some may be omitted if it is e.g. a
small bug fix and not a new feature or design change):

* **What** does the change do?  (description of the new features, changes,
  or refactorings)
* **Where** should the reviewer start the review? (what is the central
  module that changed, etc.)
* What is the **use case**, if it is a new feature?
* give an **example** of use or screenshot/plot if applicable
* Are any **new dependencies** required? (dependencies should be kept to a
  minimum, so all new dependences need to be accepted by management)
* is there a relevant **issue** open that this addresses? (use the
  #ISSUENUMBER syntax to link it)


Note that you can include syntax-highlighted code examples by using 3 back-tics:

.. code-block:: none
		
   ```python
   
   code here
   
   ```

Keep in mind
------------

* make sure you remember to update the **documentation** as well as the code!
  (see the `docs/` directory), and make sure it builds with no errors (`make
  doc`)

* Pull requests that cause tests to fail on *Travis* (the continuous
  integration system) will not be accepted until those tests pass.


Accepting a Pull Request
------------------------

`ctapipe` maintainers must do a *code review* before accepting any
pull request. During the review the reviewer can ask for changes to be
made, and the requester can simply push them to the branch associated
with the request and they will automatically appear (no new pull
request needed).  The following guidelines should be used to
facilitate the review procedure:

* Perform a Scientific or Conceptual Review if the request introduces
  new features, algorithms, or design changes
  
 - Look at the use case for the proposed change.
   
  + if the use case is missing, ask for one
  + does it make sense? Is it connected to a goal, requirement, or specification?
    
* Perform a Code Review in 2 passes
  
  - Check that all functions and classes have API documentation in the
    correct format (and check that there are no warnings generated
    whenbuilding the docs)
  - Check that there are at least basic unit tests for each function and class.
  - Run all unit tests
    
   + If any unit tests included in the pull request fail, ask the
     developer if that is normal (sometimes there are tests intended
     to fail until a feature is available or implemented)
   + If any other unit tests that previously suceeded now fail,
     **stop** and ask the developer to find out what was broken by
     their change
     
  - Read through the new code being contributed, and see that it
    follows the style guidelines and that the API
    
   + no lines over 90 cols (prefer 80, but some can go a bit over)
   + functions and variables are lower case, classes CamelCase, etc.
   + variable names give clear meaning
   + follows all other PEP8 conventions (run `pylint` or `flake8` to
     check for example), if there are obvious style problems, ask for
     them to be fixed.
     
  * Check for common coding mistakes
  * Check that the API (function and class definitions) is clear and
    easy to understand. If not, ask for it to be cleaned up
  * Check that the code uses the existing features of the ctapipe framework
  * Check that the code doesn't introduce new features that are
    already present in another form in the framework
