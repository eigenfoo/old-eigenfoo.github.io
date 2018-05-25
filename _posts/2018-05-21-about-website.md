---
title: "About This Website"
excerpt: "I recently finished putting the last bells and whistles on this
website, even though I've been writing posts for a while now. These are just
some quick notes"
tags:
    - website
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
  caption: "Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)"
last_modified_at: 2018-05-21
---

I recently finished putting the last bells and whistles on this website, even
though I've been writing posts for a bit more than a year now. These are just
some quick notes about how I made this website: maybe it'll be useful for
someone else who's trying to pitch a tent on these interwebs.

This website is hosted on [GitHub Pages](https://pages.github.com/), and the DNS
is served through [Cloudflare](https://www.cloudflare.com/). There's plenty to
say about this setup, but it's all been said before, so
[here](https://www.toptal.com/github/unlimited-scale-web-hosting-github-pages-cloudflare)
[you go](https://blog.cloudflare.com/secure-and-fast-github-pages-with-cloudflare/).
I'm also using Michael Rose's Jekyll template,
[minimal-mistakes](https://github.com/mmistakes/minimal-mistakes), so I didn't
have to code everything from the ground up. This means that my entire website is
static... well, except for the comments. I gave
[Staticman](https://staticman.net/) a shot, which probably would've been nice to
have on some spiritual (?) level ("100% static! Woohoo!") but it wasn't working
for whatever reason, so I just gave up. I use [Disqus](https://disqus.com/) for
comments now. I don't think I've spent enough time with either of these
providers to really form a good opinion, but Disqus seems to be doing its job
just fine.

All in all, I'm pretty happy with how smoothly this stuff works!

Almost everything in this thing is free. GitHub hosts my static site for free,
and Cloudflare serves my DNS for free. All I actually pay for is my custom
domain name, which costs around two cups of coffee per year. (Rest assured I'm
also squeezing
[other value](https://lifehacker.com/5881321/eight-clever-things-you-can-do-with-your-underused-personal-domain-name)
out of my domain name, too).

The biggest drawback I see to this is [GitHub's limits on size and
bandwidth](https://help.github.com/articles/what-is-github-pages/) (1GB limit on
source repositories, 100GB/hr limit on bandwidth, 100/hr limit on page builds,
etc.). But for such a small website like mine, this seems perfectly fine,
especially considering how cheap it is: I don't expect my repository to take up
more than 1GB, and I certainly don't expect to get more than, say, 100 visitors
per month.

If I wanted to get fancier, I'd probably move my stuff to [Digital
Ocean](https://www.digitalocean.com/), which is ~*~ more legit ~*~ and has some
pretty good services for reasonable prices.

So there you go! I think this stack is fantastic for my needs, and the result is
right in front of you: you're reading it right now! If you'd like, here's a link
to the [source of this website on GitHub](https://github.com/eigenfoo/eigenfoo.xyz).
Hope this helped!
