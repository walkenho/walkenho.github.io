---
title: >-
    Excuse-Me Streamlit Cloud, You Want WHAT???
header:
  overlay_image: /images/shocked_monkey.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo by [**Jamie Haughton**](https://unsplash.com/@haughters) on [Unsplash](https://unsplash.com/photos/Z05GiksmqYU)"
  actions:
    - label: ""
classes: wide
tags:
  - streamlit
  - web deployment
  - rant
excerpt: Why I won't deploy streamlit applications from my GitHub account and neither should you
---

I love [Streamlit](https://streamlit.io/), I really do. It is ridiculously easy to use and gets you where you want to be really fast. And since its execution flow is straight-forward top-to-bottom and everything gets re-run every time, there is no need to worry about what you can and cannot updated and when (in contrast to Ipywidgets+Voila). 

So what's the fuss about?

Having just finished my [meeting-organizer app](https://github.com/walkenho/meeting-attendance-organizer), I was looking for a place to deploy it. With it being written using Streamlit, an obvious choice would be [Streamlit Community Cloud](https://streamlit.io/cloud), which allows you to deploy Streamlit applications directly from your GitHub account.  

The only caveat: In order to do so, you have to  give it **full read and write access to all of your GitHub repositories, public as well as private (!)**.

_Say again???_

Now, apparantly this is not Streamlit's fault, but [an issue with GitHub itself](https://discuss.streamlit.io/t/github-permissions-too-onerous/22094), so nothing that they can really fix. However, call me paraonoid, but I won't give a third party read and write access to my all my repositories.

So what are potential solutions here? At this point, [options are two](https://www.tor.com/2008/10/30/not-only-science-fiction-but-more-science-fictional-than-anything-else-rosemary-kirsteins-steerswoman-books/):

1. Don't use Streamlit Cloud Community.

   Deploy your app somewhere else instead. Free options include [Hugging Face Spaces](https://huggingface.co/spaces), where you can either directly deploy a Streamlit space or you can deploy a Docker container which contains your Streamlit application (if you choose the latter, remember to re-route the port!). More about how to make this a smooth workflow in a post to come.
2. Do use Streamlit Cloud Community, but create a additional GitHub repository exclusively for your Streamlit deployments.

    Create a new GitHub account, which you exclusively use to deploy your Streamlit apps. If you use your GitHub profile to show-case your portfolio and therefore want to keep all of your projects together and associated to you, you can keep the development repository in your original repository and fork a copy from your deployment repository, which you then deploy to Streamlit.  

Which one to prefer? Up to you really. Just not option 0. Just don't give Streamlit access to all your private GitHub repositories. 