# How to Contribute

## Your First Pull Request
We use github for our codebase. You can start by reading [How To Pull Request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).

## Branch Organization
We use [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) as our branch organization, as known as [FDD](https://en.wikipedia.org/wiki/Feature-driven_development)

## Bugs
### 1. How to Find Known Issues
We are using [Github Issues](https://github.com/BabitMF/bmf/issues) for our public bugs. We keep a close eye on this and try to make it clear when we have an internal fix in progress. Before filing a new task, try to make sure your problem doesn't already exist.

### 2. Reporting New Issues
It is recommended that you keep the title concise yet clear. In the main body, you should elaborate on the issue you discovered, including the BMF version you are using (e.g., v0.0.12) and the complete code required to reproduce the issue. If you can also provide your thoughts on the issue's root cause, that would be even better.

### 3. Security Bugs
Please do not report the safe disclosure of bugs to public issues. Contact us by [Support Email](mailto:conduct@BabitMF.io)

## How to Get in Touch
- [Email](mailto:conduct@BabitMF.io)
- [Feishu Group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=4cev1bee-4d94-42c8-972b-4ae4a12c9da1)

## Submit a Pull Request
Before you submit your Pull Request (PR) consider the following guidelines:
1. Search [GitHub](https://github.com/BabitMF/bmf/pulls) for an open or closed PR that relates to your submission. You don't want to duplicate existing efforts.
2. Please submit an issue instead of PR if you have a better suggestion for format tools. We won't accept a lot of file changes directly without issue statement and assignment.
3. Be sure that the issue describes the problem you're fixing, or documents the design for the feature you'd like to add. Before we accepting your work, we need to conduct some checks and evaluations. So, It will be better if you can discuss the design with us.
4. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the BabitMF/bmf repo.
5. In your forked repository, make your changes in a new git branch:
    ```
    git checkout -b my-fix-branch develop
    ```
6. Create your patch. See our [Patch Guidelines](#patch-guidelines) for details

7. Push your branch to GitHub:
    ```
    git push origin my-fix-branch
    ```
8. In GitHub, send a pull request to `bmf:master` with a clear and unambiguous title.

## Contribution Prerequisites
- You are familiar with [Github](https://github.com)
- Maybe you need familiar with [Actions](https://github.com/features/actions)(our default workflow tool).

## Patch Guidelines
- If you add some code inside the framework, please include appropriate test cases
- Commit your changes using a descriptive commit message that follows [AngularJS Git Commit Message Conventions](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit).
- Do not commit unrelated changes together.
- Cosmetic changes should be kept in separate patches.
- Follow our [Style Guides](#code-style-guides).

## Code Style Guides

- See [C++ Code clang-format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format).
After clang-format installed(version 16.0.6, can be installed by: pip install clang-format==16.0.6), using command:
    ```bash
    clang-format -sort-includes=false -style="{BasedOnStyle: llvm, IndentWidth: 4}" -i <your file>
    ```
- Python source code, here use:
    ```bash
    pip install yapf==0.40.1
    yapf --in-place --recursive --style="{indent_width: 4}"
    ```


