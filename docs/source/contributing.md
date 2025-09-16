# How to Contribute

1. Create a Branch
Before you start making changes, create a new branch. Use a descriptive name that reflects the purpose of your changes (e.g., bugfix/fix-login-error, feat/add-dark-mode).

```
git checkout -b your-branch-name
```
2. Make Your Changes
Once you've made your changes, be sure to:

Add tests for any new features or bug fixes.

Follow the coding style of the project. We use [ruff] to enforce a consistent style. You can run the linter to check by:

```
ruff .
```

Update the documentation if your changes require it.

3. Commit Your Changes
Write a clear and concise commit message. We follow the Conventional Commits standard, which helps us generate changelogs and understand the history of the project.

```
git add .
git commit -m "feat: add user authentication"
```

4. Push and Open a Pull Request
Push your branch to your forked repository:

```
git push origin your-branch-name
```

Then, go to the original repository on GitHub and open a pull request (PR). In your PR description, please include:

A clear and concise summary of the changes.

Any relevant issue numbers (e.g., "Closes #123").

Steps to reproduce the bug or an example of the new feature.

## Review Process
Once you open a pull request, a maintainer will review your changes. They may leave comments asking for clarification or suggesting improvements. Don't be discouraged—this is a normal part of the open-source collaboration process!

Once your PR has been approved, a maintainer will merge it into the main branch.

Thank you again for your contribution! We appreciate your help in making this project better.