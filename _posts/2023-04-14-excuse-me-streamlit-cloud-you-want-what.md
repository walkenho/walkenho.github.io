---
title: >-
    Why You Shouldn't Deploy Streamlit Applications from Your Main GitHub Account and What You Should Do Instead
header:
  overlay_image: /images/shocked_monkey.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo by [**Jamie Haughton**](https://unsplash.com/@haughters) on [Unsplash](https://unsplash.com/photos/Z05GiksmqYU)"
  actions:
    - label: ""
classes: wide
tags:
  - Streamlit
  - web deployment
  - data privacy
excerpt: 
    How to keep your private GitHub repositories private
---

I love [Streamlit](https://streamlit.io/). It is ridiculously easy to use and allows you to code up a running front-end in basically no time at all. And since its execution flow is straight-forward top-to-bottom and everything gets re-run every time, there is no need to worry about what you can and cannot updated and when (in contrast to using Ipywidgets+Voila).

So what's the fuss about?

Having just finished my [meeting-organizer app](https://github.com/walkenho/meeting-attendance-organizer), I was looking for a place to deploy it. With it being written using Streamlit, an obvious choice would be [Streamlit Community Cloud](https://streamlit.io/cloud), which allows you to deploy Streamlit applications directly from your GitHub account.  

The only caveat: In order to do so, you have to  give Streamlit **full read and write access to all of your GitHub repositories, public as well as private (!)**.

_Say again???_

Now, apparently this is not Streamlit's fault, but [an issue with GitHub itself](https://discuss.streamlit.io/t/github-permissions-too-onerous/22094), hence there isn't really anything that Streamlit can do about it. However, call me paranoid, but I am not comfortable giving third parties read and write access to each and every single one of my private repositories.

So what are potential solutions here? At this point, [options are two](https://www.tor.com/2008/10/30/not-only-science-fiction-but-more-science-fictional-than-anything-else-rosemary-kirsteins-steerswoman-books/):

1. Don't use Streamlit Cloud Community, but deploy your app somewhere else instead.

    Free options include [Hugging Face Spaces](https://huggingface.co/spaces), where you can either directly deploy a Streamlit space or you can deploy a Docker container containing your Streamlit application (if you choose the latter, remember to re-route the port!). If you want to keep your repository in your GitHub, you can make this a smooth workflow by adding Hugging Face as a second remote.

2. Do use Streamlit Cloud Community, but create an additional GitHub account exclusively for your Streamlit deployments.

    This is the option that Streamlit Developer Relations [recommends to its corporate users](https://discuss.streamlit.io/t/github-permissions-too-onerous/22094/2). In case you use your GitHub profile as a portfolio and therefore want to keep all of your projects together and associated to you, you can keep the development repository in your original repository and fork a copy from your deployment account, which you then proceed to deploy to Streamlit.  

Now which of these is the better option? In my mind, they are much of a muchness. Both are slightly inconvenient, but none much more than the other. Hence, chose whatever fits better into your personal workflow. Personally, for my Meeting Organizer project, I went with latter.

Happy deploying!
