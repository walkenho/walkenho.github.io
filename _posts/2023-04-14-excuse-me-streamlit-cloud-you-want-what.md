---
title: >-
    Why I won't deploy Streamlit applications from my main GitHub account and neither should you
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
excerpt: 
    Excuse-Me Streamlit Cloud, You Want WHAT???
---

I love [Streamlit](https://streamlit.io/). It is easy to use and gets you what you want (i.e. a running app) really quickly. And since its execution flow is straight-forward top-to-bottom and everything gets re-run every time, there is no need to worry about what you can and cannot updated and when (in contrast to Ipywidgets+Voila).

So what's the fuss about?

Having just finished my [meeting-organizer app](https://github.com/walkenho/meeting-attendance-organizer), I was looking for a place to deploy it. With it being written using Streamlit, an obvious choice would be [Streamlit Community Cloud](https://streamlit.io/cloud), which allows you to deploy Streamlit applications directly from your GitHub account.  

The only caveat: In order to do so, you have to  give Streamlit **full read and write access to all of your GitHub repositories, public as well as private (!)**.

_Say again???_

Now, apparently this is not Streamlit's fault, but [an issue with GitHub itself](https://discuss.streamlit.io/t/github-permissions-too-onerous/22094), so nothing that Streamlit can really fix. However, call me paranoid, but I won't give a third party read and write access to my all my repositories.

So what are potential solutions here? At this point, [options are two](https://www.tor.com/2008/10/30/not-only-science-fiction-but-more-science-fictional-than-anything-else-rosemary-kirsteins-steerswoman-books/):

1. Don't use Streamlit Cloud Community, but deploy your app somewhere else instead.

    Free options include [Hugging Face Spaces](https://huggingface.co/spaces), where you can either directly deploy a Streamlit space or you can deploy a Docker container containing your Streamlit application (if you choose the latter, remember to re-route the port!). More about how to make this a smooth workflow in a post to come.

2. Do use Streamlit Cloud Community, but create an additional GitHub account exclusively for your Streamlit deployments.

    This is the option that Streamlit Developer Relations [recommends to its corporate users](https://discuss.streamlit.io/t/github-permissions-too-onerous/22094/2). In case you use your GitHub profile as a portfolio and therefore want to keep all of your projects together and associated to you, you can keep the development repository in your original repository and fork a copy from your deployment account, which you then proceed to deploy to Streamlit.  

Which one to prefer? They are both a little inconvenient, but in similar amounts, so up to you really. Just as long as you choose one of them and don't give Streamlit access to all your private GitHub repositories.
