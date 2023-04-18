---
title: >-
    Building and Deploying a Meetup Organizer App for London's PyData Meetup
header:
  overlay_image: /images/meeting-attendance-organizer-app.png
  overlay_filter: 0.1 # same as adding an opacity of 0.5 to a black background
  caption: "Screenshot of the Meeting Attendance Organizer App starting page"
  actions:
    - label: ""
classes: wide
tags:
  - streamlit
  - PyData
excerpt: 
    Why and How I built an App to Help Me Organize London's PyData Meetup
---

I recently built and deployed the [Meeting Attendance Organizer App](https://meeting-attendance-organizer.streamlit.app/). This app fullfills a simple need: It lets users upload one or more lists of names of people attending a meeting and allows them to perform one or more of the following tasks:

* Split attendee names into first name and surname.
*  Compare two lists of attendees with each other and see who is new on the second list.
* Find people in a list by either searching for their complete names or parts of their name
* Write any of the results back out, so you can share it with others.

This post explains the reason why I built it and also gives some quick insights into the how.

## Why did I built the Meeting Organizer App?

At the beginning of 2023 I joined the organizing committee of [London's PyData Meetup](https://www.meetup.com/pydata-london-meetup/). Before each Meetup, there are a few things that we organizers need to do. These include:

* A few days before the Meetup, we send a list of attendees to the building security. This file lists all attendee names, split into first names and surnames.
* At the day of the event, we send a list with all attendees that have signed up since the first list was produced.
* To make sure that we don't end up without speakers, we separately check that the speakers' names are indeed on the list!

Whilst all of this is in pricinciple straight forward, there are a few complications:

* Meetup provides a list of attendee names, but these names are the attendees' profile names. In order to be send, the names need to be split into first names and surnames.
* Whilst Meetup does provide the dates when attendees signed up for the event, it only provides the data part, not the full timestamp. Hence, we cannot use this information to generate the list updates, but need to generate the updates by comparing the first and the final version of the list of attendees.
* The Meetup page does have a search function, but the large number of group members (we currently have more than 11,000 members) renders the interface rather sluggish.

With this in mind, I wanted to build something that would allow me to easily perform the three tasks listed without having to spin up my own software environment each time. I also wanted to create something that would allow the other organizers to do the same without the need of sharing code with them.

## How did I built the App?

The app itself is fully built in Python. For this, I considered two options: [Voila Notebooks](https://github.com/voila-dashboards/voila) in combination with [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/) on one hand vs [Streamlit](https://streamlit.io/) on the other. In the end, I decided to use Streamlit over Voila. This is because the app requires users to upload data files and some of the functionality of the app then depends on the properties of the uploaded data. For this type of of use-case, I prefer Streamlit's straightforward top-to-bottom excecution over having to consider the different execution layers of the noteboook/widget combination.

Once built, I decided to host the app on Streamlit Cloud Community, which can be done directly from GitHub. In order to [protect my private GitHub repositories]() I created a second GitHub profile for the sole purpose of deploying Streamlit apps and forked my original project to this repository.

And that's where it lives now. You can see and use the app at https://meeting-attendance-organizer.streamlit.app and I hope you will find it useful!