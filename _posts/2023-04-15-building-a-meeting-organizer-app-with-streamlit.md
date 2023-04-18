---
title: >-
    Building and Deploying a Meetup Organizer App Using Streamlit
header:
  overlay_image: /images/meeting-attendance-organizer-app.png
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Screenshot of the Meeting Attendance Organizer App starting page"
  actions:
    - label: ""
classes: wide
tags:
  - streamlit
  - PyData
excerpt: 
    Why I build a Meetup Organizer App for London's PyData Meetup
---

## Why?

At the beginning of 2023 I joined the organizing committee of [London's PyData Meetup](https://www.meetup.com/pydata-london-meetup/). Before each Meetup, there are a few things that we organizers need to do. These include:

* A few days before the Meetup, we send a list of attendees to the building security. This file lists all attendee names, split into first names and surnames.
* At the day of the event, we send a list with all attendees that have signed up since the first list was produced.
* To make sure that we don't end up without speakers, we separately check that the speakers' names are indeed on the list!

Whilst all of this is in pricinciple straight forward, there are a few complications:

* Meetup provides a list of attendee names, but these names are the attendees' profile names. In order to be send, the names need to be split into first names and surnames.
* Whilst Meetup does provide the dates when attendees signed up for the event, it only provides the data part, not the full timestamp. Hence, we cannot use this information to generate the list updates, but need to generate the updates by comparing the first and the final version of the list of attendees.
* The Meetup page does have a search function, but the large number of group members (we currently have more than 11,000 members) renders the interface rather sluggish.

With this in mind, I build an app, that I could host on the internet that would allow any of the Meetup organizers to easily perform any of the three tasks mentioned above.

## How?

The app itself is fully built in Python. For this, I considered two options: [Voila Notebooks](https://github.com/voila-dashboards/voila) in combination with [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/latest/) on one hand vs [Streamlit](https://streamlit.io/) on the other. In the end, I decided to use Streamlit over Voila. This is because the app requires users to upload data files and some of the functionality of the app then depends on the properties of the uploaded data. For this type of of use-case, I prefer Streamlit's straightforward top-to-bottom excecution over having to consider the different execution layers of the noteboook/widget combination.

Once built, I decided to host the app on Streamlit Cloud Community, which can be done directly from GitHub. In order to [protect my private GitHub repositories]() I created a second GitHub profile for the sole purpose of deploying Streamlit apps and forked my original project to this repository.

And that's where it lives now. You can see and use the app at https://meeting-attendance-organizer.streamlit.app and I hope you will find it useful!