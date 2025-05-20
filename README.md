# research_project
Advanced ML research project

## Managing Project Repository

After cloning the repository to your local machine, all work must be done in your own branch. Create your branch by typing the following in your terminal:
``` git checkout -b <name>-branch ```
Replace `<name>` with your name.

>[!IMPORTANT]
>1. Do not attempt to work on the code in the main branch.
>2. Do not merge your code into the main branch on your local machine.
>3. Only commit and push your own branches onto the GitHub repo and submit a pull request.

To commit and push your code onto your branch use the following commands in the terminal:
```
  # If you want to add all modified files use:
  git add .
  git commit -a -m "your commit message"

  # Otherwise use:
  git add <filename-1>.....<filename-n>
  git commit -m "your commit message"

  # This code is necessary in both cases
  git push origin <your-branch>
```

Once pushed, submit a pull a request and I will merge your code into main. After the code is merged i will notify you and you must pull the changes into your machine:
```
  git switch main
  git pull
```

If you wish to merge main with your own branch so that your branch gets the latest code you may type the following:
```
  git switch <your-branch>
  git merge main

  // OR

  git switch <your-branch>
  git pull origin main
```
Always make sure that your branch has the latest code.

