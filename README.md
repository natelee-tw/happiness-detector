# Introduction

A bare minimum basic template for all 100E projects.

Run `source ./init.sh` and customise the folders as required.

Checkout [cookie cutter data science](https://drivendata.github.io/cookiecutter-data-science)
for ideas on how you might want to further customise
the project.

Write an introduction to your project in this 
section and describe what it does.

Create sub sections below to clearly document how
to build, run, test, deploy, etc.  

The [scripts](scripts) folder contains useful scripts to help
run model training on external infrastructure and
build/package your model.


## Polyaxon setup

### Configuration

Ensure you have the polyaxon client installed.

#### Install Polyaxon client and configure it
```
pip install polyaxon-cli==0.5.6

# Configure your machine to use our Polyaxon cluster
polyaxon config set --host=polyaxon.okdapp.tekong.aisingapore.net --port=80
polyaxon login -u <username>
```

#### Create and initialise your polyaxon project

**Replace {your_project_name} below with the actual project name**
  
```
polyaxon project create --name={your_project_name} --private true
polyaxon init {your_project_name}
```

#### Start your notebook

```
polyaxon upload
polyaxon notebook start -f polyaxon/notebook.yml
```

Wait around a couple of seconds and navigate to 
the url printed on the console.

You can also open up the dashboard at:

http://polyaxon.okdapp.tekong.aisingapore.net

Navigate to your project and you should see a 
little **green notebook** link above the Readme on 
the dashboard.

Click on it and your jupyter server interface should
show up.

## Code setup (git)

1. Open a terminal from your notebook interface

2. If you don't see a folder with the same name as your polyaxon project.
   Open up a terminal and manually run the script 
   
   ```
   link_workspace.sh
   ```
    
   at the root of your jupyter server.
   
   Now, clone your repo in the linked workspace folder

    **Replace the following in the commands below accordingly:** 
    - {some name to identify your workspace}
    - "http://your-repo"    
    
    ```
    git clone "http://your-repo"
    ```

3. Configure your git repo in polyaxon 

    **Replace the following in the commands below accordingly:**
    - xxxx@aiap.sg
    - your name
    - rest of path to your_git_repo
     
    ```
    cd {your cloned repo}
    git config user.email "xxxx@aiap.sg"
    git config user.name "your name"
    ```

4. Add your username to the repo url.

    ```
    vi .git/config 
    ``` 
    Add your **username with @** after *http://* under remote origin
    url \
    Eg.
    ```
    [remote origin]
      url = http://my_username@gitlab.int.aisingapore.org/aisg/polyaxon-examples.git
    ```

### Conda env updates

1. For dependency management, manually add any new libraries required to
the **pip section** of [conda.yml](conda.yml). Only add to the **conda
dependencies section** if strictly required. This is to help reduce the
conda dependency search which is slow, error prone and can cause issues
when building on gitlab.   

### Uploading files to your polyaxon data

Use the jupyter upload function to upload. You need to ensure it 
goes into your project workspace folder or the **data** folder in
your jupyter server root so that it gets persisted. Anything other 
than these locations will not be persistent and will disappear if 
your jupyter server is shutdown or crashes.


## End Notes

- If container gets killed/stopped, just restart the notebook.
- Check build and statuses/logs in Polyaxon dashboard 
for any startup failures
- Refer to [polyaxon-examples repo][1] for more 
troubleshooting tips

<!-- Reference links -->

[1]: http://gitlab.int.aisingapore.org/aisg/polyaxon-examples 