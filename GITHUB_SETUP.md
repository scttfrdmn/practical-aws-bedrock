# GitHub Repository Setup Instructions

Follow these steps to create your GitHub repository and push the code:

## 1. Create a new repository on GitHub

1. Go to [GitHub](https://github.com/) and log in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Name your repository "practical-aws-bedrock"
4. Add a description: "A comprehensive, action-oriented guide to building generative AI applications with AWS Bedrock"
5. Choose "Public" visibility (recommended for documentation projects)
6. Do not initialize with a README, .gitignore, or license (we already have these files)
7. Click "Create repository"

## 2. Push your local repository to GitHub

After creating the repository, GitHub will display commands to push an existing repository. Use the following commands:

```bash
# Navigate to your project directory (if not already there)
cd /Users/scttfrdmn/src/aws-bedrock-inference

# Update the remote URL to point to your new GitHub repository
git remote add origin https://github.com/YOUR-USERNAME/practical-aws-bedrock.git

# Push the code to GitHub
git push -u origin main
```

Replace "YOUR-USERNAME" with your actual GitHub username.

## 3. Configure GitHub Pages

1. Go to your repository on GitHub
2. Click "Settings" at the top of the repository page
3. In the left sidebar, click "Pages"
4. Under "Source", select "GitHub Actions" as the build and deployment source
5. GitHub will automatically use the workflow file we've created

## 4. Verify the deployment

1. After pushing the code, go to the "Actions" tab in your repository
2. You should see the "GitHub Pages" workflow running
3. Once completed, GitHub will provide a URL where your site is published
4. Visit the URL to verify your site is working correctly

## 5. Additional configurations (optional)

### Custom domain
1. If you have a custom domain, you can configure it in the GitHub Pages settings
2. Add your domain name in the "Custom domain" field
3. Update your DNS settings according to GitHub's instructions
4. Add a CNAME file to your `docs` directory with your domain name

### Enable discussions
1. Go to repository settings
2. Scroll down to the "Features" section
3. Check the box next to "Discussions"
4. This allows readers to ask questions and provide feedback

## Next steps

After setting up the GitHub repository:

1. Continue developing content according to the learning path
2. Monitor GitHub Actions for any deployment issues
3. Consider setting up Google Analytics to track usage (update `_config.yml`)
4. Regularly commit and push updates to keep the content fresh

Your project is now set up for collaborative development and public access through GitHub Pages!