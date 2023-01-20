# How to Create a Computer Vision Model to Save the Ugly Duckling from a Youth of Misery in Less Than 30 Lines of  Code

## Introduction

No New Year without its resolution. So how about diving into computer vision, learning a new library that allows you to train a computer vision model in less than 30 lines of code and revisiting some good old fairy tales all at once?

Hans Christian Andersen's [tale of the ugly duckling](https://americanliterature.com/author/hans-christian-andersen/short-story/the-ugly-duckling) tells us about a cygnet (i.e. a baby swan) brought up as part of a family of ducks and being bullied by its siblings and everybody else for being ugly (read *different to them*). If only the poor cygnet had had somebody to tell it that it in fact was a swan instead! Entering Our Hero, Machine Learning to the rescue!

In this project, we will learn how to use fastai, Gradio and HuggingFace to build a magic mirror that can distinguish a duckling from a cygnet in four main steps. We will learn how to:

1. Create a dataset from scratch, using a duckduckgo image search.
2. Train a train a computer vision model using [fastai](https://www.fast.ai/).
3. Build a magic mirror app using [Gradio](https://www.gradio.app/).
4. Deploy the magic mirror on [HuggingFace](https://huggingface.co/).

To follow along, find the complete code in [my GitHub respository Tales-of-1001-Data]().

So let's get started, so that the ugly duckling can finally ask the magical question

**"Mirror, mirror on the wall, am I a duckling or a cygnet after all?"**

<img src="mirror-mirror.png" alt="**Mirror, mirror on the wall, am I a duckling or a cygnet after all?**" width="15%" align="middle"/>

## Preparations

### Installing Libraries

The training part of this notebook requires two non-standard libraries:
* We use `duckduckgo_search` to search for cygnet and duckling images to create the dataset. 
* We use `fastai` for training the computer vision model. [fastai](https://docs.fast.ai/), a higher-level interface to PyTorch, allows users to train state-of-the-art deep learning models in very few lines of code with the shortest being [four lines of code from zero to image-classifier](https://www.fast.ai/posts/2021-08-02-fastdownload.html).

I have run this notebook on Google Colab (because yeah, free GPUs!), which comes with fastai pre-installed, so the only thing required to install is `duckduckgo_search`. If you are running this somewhere else, you might have to install the `fastai` library, too. At the time of writing, `mamba` is the recommended way of doing so, however check the [fastai homepage](https://docs.fast.ai/) for up-to-date instructions. Let's go ahead and install `duckduckgo_search`.


```python
! pip install -U duckduckgo_search
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting duckduckgo_search
      Downloading duckduckgo_search-2.8.0-py3-none-any.whl (34 kB)
    Collecting requests>=2.28.1
      Downloading requests-2.28.1-py3-none-any.whl (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.8/62.8 KB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting click>=8.1.3
      Downloading click-8.1.3-py3-none-any.whl (96 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.6/96.6 KB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.28.1->duckduckgo_search) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.28.1->duckduckgo_search) (2022.12.7)
    Requirement already satisfied: charset-normalizer<3,>=2 in /usr/local/lib/python3.8/dist-packages (from requests>=2.28.1->duckduckgo_search) (2.1.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.28.1->duckduckgo_search) (2.10)
    Installing collected packages: requests, click, duckduckgo_search
      Attempting uninstall: requests
        Found existing installation: requests 2.25.1
        Uninstalling requests-2.25.1:
          Successfully uninstalled requests-2.25.1
      Attempting uninstall: click
        Found existing installation: click 7.1.2
        Uninstalling click-7.1.2:
          Successfully uninstalled click-7.1.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    flask 1.1.4 requires click<8.0,>=5.1, but you have click 8.1.3 which is incompatible.[0m[31m
    [0mSuccessfully installed click-8.1.3 duckduckgo_search-2.8.0 requests-2.28.1


### Importing Modules

Library imports should happen at the top of your code, so as always let's start with importing the required libraries.


```python
from pathlib import Path
from time import sleep
from dataclasses import dataclass
from typing import List

from duckduckgo_search import ddg_images

from fastai.data.transforms import get_image_files, parent_label, RandomSplitter

from fastai.interpret import ClassificationInterpretation

from fastai.metrics import error_rate

from fastai.vision.augment import Resize, RandomResizedCrop, aug_transforms
from fastai.vision.core import DataBlock
from fastai.vision.data import CategoryBlock, ImageBlock
from fastai.vision.all import vision_learner
from fastai.vision.utils import download_images, get_image_files, verify_images

from fastcore.foundation import L

from torchvision.models.resnet import resnet18
```

Note that the recommended way of importing fastai libraries is to use

`from fastai import *`

or at least

`from fastai.vision.all import *`.

This is partially due to the [large number of required imports](https://forums.fast.ai/t/cnn-learner-returning-a-sequential-object-with-missing-methods/90378) and partially due to the [extensive monkey patching in the library](https://forums.fast.ai/t/cnn-learner-returning-a-sequential-object-with-missing-methods/90378). To get an impression of how a one-by-one import looks like (including comments on patched functions), take a look at [walk-with-fastai's first lesson](https://walkwithfastai.com/Pets).

However, since in software engineering, wildcard imports are typically dicouraged (with exceptions for ad-hoc work), I import the needed libraries one by one and resolve [issues](https://stackoverflow.com/questions/65128126/fast-ai-attributeerror-learner-object-has-no-attribute-fine-tune) as they appear. This also forces me to become more familiar with the internal organization of the library itself, which I appreciate. When running into issues I find the following steps helpful.

### How to troubleshoot imports

1. Run `from fastai import *` to see if this solves your issue.
2. If so, use `which functionname` to find out from where you should be actually importing your library
3. Adapt your import statement accordingly. 

An example where this has helped me was in finding out that `vision_learner` should be imported from `fastai.vision.all` instead of from `fastai.vision.learner`.
If you don't want to go into too many details with your imports, you can also import the modules from `fastai.vision.all` (e.g. `from fastai.vision.all import RandomSplitter`).

Finally, in addition to the data crunching and general learning modules, we import the resnet18 model which is the architecture we will be using for our learning task. More on resnet18 below.

## Creating the Dataset

No machine learning without data. To create our dataset, we implement the following steps:

* Use duckduckgo's image search to get a list of urls of images of ducklings and cygnets.
* Download the images to our system using fastai's `download_images()`.
* Check if the downloaded images are ok using fastai's `verify_images()`.
* Delete images which are not ok.

Once we have built the dataset, we will use it to train a classifier. A common image classification folder structure is to save the images in folders named according to their label. To decouple the label from the actual searchterm and create more flexibility in setting up our search, we create a `Searchterm` dataclass that holds both the search term and the label.


```python
IMAGEPATH = Path().cwd()/'images'

@dataclass
class Searchterm:
  searchstring: str
  label: str


def search_images(searchterm: str, max_results: int) -> L:
    """Uses duckduckgo to search for images"""

    return L(ddg_images(searchstring,
                        max_results=max_results)).itemgot('image')


def delete_invalid_images(imagepath: Path) -> int:
    """Deletes invalid images in imagepath"""

    invalid_images = verify_images(get_image_files(imagepath))
    invalid_images.map(Path.unlink)
    return len(invalid_images)


def download_search_data(terms: List[Searchterm], n_searches) -> None:
    """Searches for images and downloads results"""
    for term in terms:
        print(f"Downloading {n_searches} images for {term.searchstring} into folder {term.label}.")
        folder = IMAGEPATH/term.label
        folder.mkdir(parents=True, exist_ok=False)

        download_images(folder,
                        urls=search_images(f"{term.searchstring}",
                                        max_results=n_searches))

        print(f"Deleted {delete_invalid_images(IMAGEPATH/term.label)} invalid images for {term.searchstring}")
```

### On Finding the Right Search Terms

To ensure that you download a useful dataset, it helps to start the process by manually running a few searches and having a look at the images you find with your query strings. Doing this process manually allows you to quickly tweak your query string so that it returns useful images and avoid the [garbage in - garbage out problem](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out).

For this project, I found that searching for "cygnet" not only found images of actual cygnets, but also returned images of a brand of the same name, which we don't want to include in our data set. 

![Not the type of cygnet we were looking for](wrong-type-of-cygnets.png)

There are technical ways of dealing with this problem, but the most straight forward solution is to simply change the search string to "baby swan", which solved the issue perfectly. In a similar project, I wanted to get photos in different lighting conditions, but realized that whilst looking for "objectname sun" returned useful results, looking for "objectname shade" did mostly return undesired results, so was not recommendable. These issues can be addressed a lot easier by doing a quick manual search upfront than by cleaning up your data set afterwards.

Once we have found good search terms, we can run our search and download some images! Remember to start from a clean directory structure to avoid potential issues with unbalanced data sets (in case you have run your code before, but interrupted it for some reason or similar). 


```python
# Clean folder structure
IMAGEPATH.delete()

download_search_data([Searchterm(searchstring='baby swan', label='cygnet'),
                      Searchterm(searchstring='duckling', label='duckling')],
                     n_searches=100)
```

    Downloading 100 images for baby swan into folder cygnet.
    Deleted 3 invalid images for baby swan
    Downloading 100 images for duckling into folder duckling.
    Deleted 2 invalid images for duckling


Nice! 

### Training the Image Classifier

Now that we have some data, we are ready to train the classifier. The first step is to convert the images in their folders into a structure that fastai models, which are called learners, can understand. fastai learners expect the data to come in the shape of a `DataLoaders` object. One way to create a `DataLoaders` object is to specify a `DataBlock` first, then call the `dataloaders()` method on it. This is how this looks like for our case.


```python
birds = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_items=get_image_files,
                  get_y=parent_label,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42),
                  item_tfms=[RandomResizedCrop(256, min_scale=0.8)],
                  batch_tfms=aug_transforms())

dls = birds.dataloaders(IMAGEPATH, bs=64)
```

Let's look a bit closer at the inputs to the `DataBlock`.

The five questions you need to answer when creating a `DataLoaders` object are:

* What type of data is the input data? What type of data is the output data?
* How do we load the data?
* For a classification task: How do we find the data labels?
* How do we create a validation set?
* Which data transformations do we apply? Are they individual or batch transformations?

  (Data transformations, which will be applied before the data is fed into the training algorithm, come in two flavours, individual and batch transformations. Individual transformations are, as the name implies, run for each image individually, whilst batch transformations are applied to a whole batch in parallel at the same time, making them significantly faster.)


From the code above, we can find the answers to our questions.

* `blocks=(ImageBlock, CategoryBlock)`

   Train an image classifier. Use images as input and output categories. 
* `get_items=get_image_files`
    
   Load the images by applying the `get_image_files` function to the path specified as input to the `dataloaders` function.
* `get_y=parent_label`
   
   Get the labels by using the `parent_label` function, which extracts the label from the folder name.
* `splitter=RandomSplitter(valid_pct=0.2, seed=42)`

  Split the data into training and validation sets by randomly assigning 20% of the data to the validation set. Seed the random number generator to 42. Note that `Dataloaders` forces you to adhere to best practice and create a validation set. Here, we seed the random number generator to make the split reproducible across multiple runs. As always, which seed you choose is completely up to you (even though [42 is often used](https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a)).

#### Data Transformations: Image Resizing and Augmentation
The final question deals with data transformations. Typical data transformations to apply to image data are resizing and image augmenation tasks. In order to feed images into a neural network, they need to be of the same size. We can achieve this by:

  * cropping
  * squishing
  * padding
  
All of these have their drawbacks. Cropping might delete important parts of the image, squishing distorts the image, padding introduces useless data thereby unnecessarily increasing computational costs. We choose

* `item_tfms=[RandomResizedCrop(256, min_scale=0.8)]`

   Randomly crop a part of the image (keep at least 80% of the original image), then scale it to 256x256 pixels. A different random crop is selected for each batch. 

In order to prevent overfitting (especially when dealing with small datasets), we might also introduce augmentations to our image dataset. Augmentations can happen on an image or on a pixel level. Examples of augmentation on an image level are:
  * flipping
  * rotation
  * perspective warping

  Examples of pixel augmentations are changes to:
  * brightness
  * saturation
  * hues
  * contrast
  
Different augmentations are applied in each batch.
  
* `batch_tfms=aug_transforms()`

  Here, we use a set of transformations designed by fastai to work well for natural photos. 

Now that we have created our ML data set (including introducing image augmentations), let's inspect some images as they might be fed into the algorithm.


```python
# Inspect some images
dls.train.show_batch(max_n=6, nrows=1)
```


    
![png](output_30_0.png)
    


Looking good :)

Next, we set up the learning part. We use fastai's `vision_learner` to train a ResNet-18 model- a [Residual Neural Network (ResNet)](https://arxiv.org/abs/1512.03385) with 18 hidden layers. ResNets include skip-layers into their architecture, which allow gradient information to propagate through the network more easily by adding the output of previous layers to layers deeper in the network. This counteracts the [problem of vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), which again allows for the training of deeper networks (the original paper demonstrated the successful training of a 152 layer network and even experimented with up to 1002 layers). The ResNet-18 architecture is the smallest ResNet architecture, making it quite fast to train. Whilst smaller networks can result in lower prediction accuracy, for our small toy problem a ResNet18 archicture is perfectly suitable.

To train our network, we use a technique called transfer learning. In transfer learning, we use weights that have been pretrained on a large corpus of data and only slightly adapt ("fine-tune") some of them (often only the weights of the actual classification layer) to our specific problem. By default, fastai uses weights which are pre-trained on the [ImageNet dataset](http://www.image-net.org).


To find a good number of fine-tuning steps (good accuracy without overfitting), we experiment with the number of steps whilst observing the training and validation loss. Whilst the training loss will continue to go down with more training steps, the validation loss will first decrease, then at some point start to increase again. The number of steps after which the validation loss starts increasing is a good number of training steps. In our case, this is 5.

So let's train the model!


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)

# 5 is a good number of steps, afterwards the validation loss starts increasing
learn.fine_tune(5)
```

    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:209: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      f"The parameter '{pretrained_param}' is deprecated since 0.13 and will be removed in 0.15, "
    /usr/local/lib/python3.7/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.322290</td>
      <td>0.665758</td>
      <td>0.243243</td>
      <td>00:56</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.537393</td>
      <td>0.333642</td>
      <td>0.135135</td>
      <td>00:58</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.390109</td>
      <td>0.203725</td>
      <td>0.081081</td>
      <td>00:58</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.283819</td>
      <td>0.161411</td>
      <td>0.054054</td>
      <td>00:58</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.218951</td>
      <td>0.160250</td>
      <td>0.054054</td>
      <td>01:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.174419</td>
      <td>0.154186</td>
      <td>0.054054</td>
      <td>00:57</td>
    </tr>
  </tbody>
</table>


### Evaluating Your Model

Now that we have a trained model, let's evaluate its performance. Loss is the algorithm's metric of driving the optimization to better results, but it's not a very humanly understandable metric. So let's look at the confusion matrix instead to see how our model is doing.


#### Confusion Matrix

The confusion matrix compares the predictions and actual class of each image and shows us the results for all images in the validation set. In the confusion matrix below, we see that all cygnets in the validation set were correctly classified as such, whilst two of the ducklings were misclassified as cygnets. Overall not too bad an outcome. 


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](output_35_4.png)
    


#### Largest Losses

For a more detailed view, let's go back to the losses. Using `interp.plot_top_losses()` function, we can display the images with the highest losses. A higher loss means that either the algorithm has correctly predicted the class of an image, but is not very certain of it or it has incorrectly predicted the class of an image (the more certain of an incorrect prediction the higher the loss). 


```python
interp.plot_top_losses(10, nrows=2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](output_37_2.png)
    


The two highest loss values are produced by the misclassified ducklings, in order of descending certainty, followed by correctly classified images, but with less certainty. 

An observation: It is noticable that the image with the highest loss is an image of a duckling that looks very different from most of the other ducklings in the dataset. My impression is that most of the ducklings in the dataset are Mallard ducklings, whilst this one is not. Unfortunately I could not find which type of duck this might be (any ornithologists in the audience willing to help out?). This is a good demonstration of why when building datasets from searches it is a good idea to save the URLs and potentially the names of the retrieved files. It makes these type of investigations a lot easier to conduct. Going back to the original task, if we were serious about good performance, it might be beneficial to deal with these dataset outliers by for example eliminating everything which isn't a Mallard duckling from our dataset (assuming that our users are only interested in classifying Mallards that is) or adding more ducklings of other types to the data. 

Assuming that you would like to delete all non-Mallard ducklings from the dataset, you can use the `interp.top_losses(items=True)` to display all losses together with the associated file paths and then use the filepaths to purge the images from your dataset.


```python
# items=True returns the file paths
interp.top_losses(items=True)
```




    (TensorBase([4.0928e+00, 1.0488e+00, 2.4337e-01, 1.5965e-01, 7.6476e-02,
                 3.3652e-02, 1.8994e-02, 1.2116e-02, 8.2581e-03, 4.4039e-03,
                 2.3067e-03, 1.1670e-03, 1.0680e-03, 3.9296e-04, 1.9465e-04,
                 1.7105e-04, 1.6032e-04, 1.5162e-04, 1.2433e-04, 1.0645e-04,
                 9.7032e-05, 8.8569e-05, 6.5563e-05, 6.4609e-05, 5.8292e-05,
                 5.4239e-05, 4.1842e-05, 3.9338e-05, 3.0517e-05, 2.6583e-05,
                 1.7047e-05, 1.4305e-05, 3.5763e-06, 1.5497e-06, 1.0729e-06,
                 4.7684e-07, 2.3842e-07]),
     TensorBase([ 2,  6, 18, 12, 32, 11,  8, 17, 31, 26,  3, 22, 14,  7, 34, 20, 24,
                 33, 36,  4, 16, 29, 21, 15, 13, 10, 35,  1, 19,  9, 23, 27, 28, 30,
                 25,  0,  5]),
     (#37) [Path('/content/images/duckling/2f13f591-286d-4f14-afd2-20aa4ba8d852.jpg'),Path('/content/images/duckling/6fce9794-6570-4be0-afd9-8b4db5eb77d0.jpg'),Path('/content/images/duckling/e0951cbb-726c-4f12-9259-6f24f7981504.jpg'),Path('/content/images/duckling/54388c76-bf64-4751-96a9-0624c0139719.JPG'),Path('/content/images/duckling/cab54506-2346-47d1-9490-bab31df5e02a.jpg'),Path('/content/images/cygnet/620949d6-bc01-40ff-a6ef-f6b45744e43d.jpg'),Path('/content/images/cygnet/95d585bd-5c02-4fa9-a3ad-fb958100f98f.jpeg'),Path('/content/images/cygnet/2f4d4c09-e296-4066-9fb6-6db37eb8ce0d.jpg'),Path('/content/images/duckling/3b9c58ba-7962-41b6-b763-cc880b5b1929.jpg'),Path('/content/images/duckling/248f87be-2c00-4e7a-8e05-9fafd0b40eb9.jpg')...])



### Other Options for Data Cleaning

Another handy tool for image data cleaning is the `ImageClassifierCleaner`, which is explained best by demonstrating its function. Just run it on your learner to get a drop-down interface allowing you to easily relabel or delete images from your dataset. 


```python
from fastai.vision.widgets import ImageClassifierCleaner

cleaner = ImageClassifierCleaner(learn)
cleaner
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    VBox(children=(Dropdown(options=('herring', 'iceland', 'lesser_black-backed'), value='herring'), Dropdown(opti…


![ImageClassifierCleaner](data-cleaning.png)

Note that the above only provides the graphical interface, you still need to relabel and/or delete the actual images. You can do this by using the following code. You need to run it for each combination of drop-downs that you want to treat.


```python
import shutil

def execute_cleaning(cleaner):
    # Run this for every combination of category and validity, it does not save!

    # delete the ones to be deleted
    for idx in cleaner.delete():
       cleaner.fns[idx].unlink()

    # move the ones to be relabeled into their new folders
    for idx, category in cleaner.change():
       shutil.move(str(cleaner.fnx[idx], IMAGEPATH/category))
```

### Export Model

Finally we can export the model for later use. In the next step, we will see how to export it to HuggingFace to deploy it!app


```python
learn.export('duckling_learner.pkl')
```

## Interface Developing and Deployment using Gradio on HuggingFace

Now that we have a trained model, we want to host it somewhere and share it with the world (and all the cygnets and ugly ducklings of course!). Nowadays, there are plenty of options to publicly host a model serving app for free. 
There are two aspects of serving an app - building the interface and hosting it. Where you can host your interface partially depends on how you decide to implement it.

Some easy, low-code (or at least purely Python) options for building graphical model interfaces are:

* Gradio
* Streamlit
* Voila

Some options to serve your interface are:

* Gradio: Hugging Face
* Streamlit: Hugging Face, Streamlit Community
* Voila: GitHub, Binder

I decided to build my app using Gradio and host it on Hugging Face Community. This was mostly because I had not used either of them before and wanted to give them a spin. Either of the options above will serve you well. This is how my app looks like. To try it out for yourselve, find its [live version on HuggingFace](https://huggingface.co/spaces/walkenho/ugly-duckling-magic-mirror).

<img src="app-screenshot.png" alt="**Gradio App on HuggingFace?**" width="75%" align="middle"/>

In order to build this and host it on HuggingFace, you need six files: 

* the main file defining the Gradio app
* the ML model for scoring
* a `README.md` defining the app's metadata (eg its title)
* a `requirements.txt` file defining extra packages that need to be installed
* a `.gitignore` file (telling HuggingFace's git which files to ignore and which to handle using `git-lfs`)
* (optional) the example images that the user can click on

You can find all the files on [my HuggingFace account](https://huggingface.co/spaces/walkenho/ugly-duckling-magic-mirror/tree/main).

### File 1: Main File - Gradio App

This is the complete code for the main Gradio App:


```python
from pathlib import Path
from fastai.vision.learner import load_learner
import gradio as gr

MODELPATH='cygnet-vs-duckling.pkl'

learn = load_learner(MODELPATH)
categories = learn.dls.vocab

def classify_image(image):
    _, _, probs = learn.predict(image)
    return dict(zip(categories, map(float, probs)))

title = 'Mirror, Mirror on the Wall, am I a Duckling or a Cygnet after all?'
description = """Hans Christian Andersen's tale of the ugly duckling tells us about the sad youth of a cygnet which is accidentally brought up in a family of ducks and is ostrized on the account of it being different. But what if the cygnet had had a magic mirror to tell it that it had been a young swan all along? Machine learning to the rescue!"""

examples = ['duckling.jpg', 'cygnet.jpg']

article = '**Technical Details**: The classification model was build using a resnet-18 architecture. Training was done using a transfer learning approach. I took the publicly available weights that were pre-trained on the ImageNet data set and fine-tuned them using about 80 images of ducklings and cygnets each.\nNote that it is binary classifier and will therefore only output "cygnet" or "duckling", "other" is not an option.' 

app = gr.Interface(fn=classify_image,
                   inputs=gr.components.Image(),
                   outputs=gr.components.Label(),
                   examples=examples,
                   title=title,
                   description=description,
                   article=article,
                   allow_flagging='never')

app.launch()
```

Easy, right? But what does it do? Let's start by looking at the `Interface` function. 

The parameters of the `Interface` function tell gradio that it is to take an image as an input and generate a label as output using the scoring function `classify_image`, which we defined above. Gradio expects the scoring function to output a dictionary with the labels as keys and the probabilities as values. 

The probabilities we get from the `predict` function called on the previously loaded model, the labels come from the `dls.vocab` attribute. One final thing to note is that `predict` returns the probabilities as tensors. This means that we need to convert the tensors to floats for gradio to be able to handle them. 

To show this, here is an example of how to use `predict` and its output format. Below, we load an image from disk, convert it into a pillow file and score it using the previously loaded model. Note how the returned probabilities are tensors? 


```python
from fastai.vision.all import PILImage

im_duckling = PILImage.create('/content/images/duckling/54388c76-bf64-4751-96a9-0624c0139719.JPG')
im_duckling.thumbnail((192, 192))
im_duckling
```




    
![png](output_55_0.png)
    




```python
learn.predict(im_duckling)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    ('duckling', TensorBase(1), TensorBase([0.0242, 0.9758]))



Now that we have the main interface, let's have a quick look at the additional files. 

### File 2: ML Model

We need to upload the previously saved model to HuggingFace. In order to do this, you need to have [Git LFS](https://www.atlassian.com/git/tutorials/git-lfs) installed and enabled in your repository. Git LFS (Large File Storage) is a Git extension that allows you download large files in your repository lazily, reducing their impact on your repository handling. 

### File 3: Metadata - Readme.md

The Readme.md contains the metadata for the app. For the app above, it looks like this.

```
title: Magic Mirror
emoji: 🪞🦢
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 3.10.1
app_file: app.py
pinned: false
```

### File 4: Dependencies - Requirements.txt

Since we told HuggingFace to use gradio in the Readme, gradio is already installed. However, we still need to install fastai, which leaves the requirements file with a single line:
```
fastai
```

### File 5: Handling Files in Git: .gitignore

The `.gitignore` file comes with the HuggingFace repository. You can use it as it is. 


### File(s) 6 (optional): Example Images

Just upload the images. Again, you need to have Git LFS installed.

And that's it! Upload everything to Hugging Face and tell all your friends about your awesome classifier! :) 

## Final Comment(s)

The model produced here is a toy model to demonstrate how to quickly build a prototype using fastai. There are many short-comings to it, including the fact that it is a binary classifier for two categories that do not span the entire space. This means that for any image that you submit, it can only decide if it thinks that the image resembles a duckling more than a cygnet or vice versa. Saying "Don't be silly, this image is clearly neither!" is just not an option. So always be suspicious of unexpected accuracy.

With that being said, I will leave you to marvel at these results that one of my friends produced with it.


<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/huey-dewey-louie.png" alt="Life is like a Hurrican..." width="50%">
    <figcaption>Huey, Dewey and Louie</figcaption>
</figure>

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/swanlake.png" alt="Swanlake Ballerina" width="50%">
    <figcaption>Clearly a Swan ;)</figcaption>
</figure>


Or if you are feeling adventurous test it out for yourself...


<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/images/jessica-duckling.png" alt="Apparantly I am a duckling..." width="50%">
    <figcaption>Apparently I am more of a duckling than Huey, Dewey and Louie together. Not sure what my self-esteem has to say to that ... :-/
</figcaption>
</figure>
