---
title: Streaming Data with Tornado and WebSockets
excerpt: "WebSockets with the Tornado web framework is a simple, robust way to
handle streaming data. I walk through a minimal example and discuss why these
tools are good for the job."
tags:
  - python
  - streaming
  - tornado
  - websocket
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background8.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2021-10-05
---

A lot of data science and machine learning practice assumes a static dataset,
maybe with some MLOps tooling for rerunning a model pipeline with the freshest
version of the dataset.

Working with streaming data is an entirely different ball game, and it wasn't
clear to me what tools a data scientist might reach for when dealing with
streaming data[^1].

I recently came across a pretty straightforward and robust solution:
[WebSockets](https://datatracker.ietf.org/doc/html/rfc6455) and
[Tornado](https://www.tornadoweb.org/en/stable/). Tornado is a Python web
framework with strong support for asynchronous networking.  WebSockets are a
way for two processes (or apps) to communicate with each other (similar to HTTP
requests with REST endpoints). Of course, Tornado has pretty good support for
WebSockets as well.

In this blog post I'll give a minimal example of using Tornado and WebSockets
to handle streaming data. The toy example I have is one app (`server.py`)
writing samples of a Bernoulli to a WebSocket, and another app (`client.py`)
listening to the WebSocket and keeping track of the posterior distribution for
a [Beta-Binomial conjugate model](https://eigenfoo.xyz/bayesian-bandits/).
After walking through the code, I'll discuss these tools, and why they're good
choices for working with streaming data.

For another tutorial on this same topic, you can check out [`proft`'s blog
post](https://en.proft.me/2014/05/16/realtime-web-application-tornado-and-websocket/).

## Server

- When `WebSocketServer` is registered to a REST endpoint (in `main`), it keeps
  track of any processes who are listening to that endpoint, and pushes
  messages to them when `send_message` is called.
  * Note that `clients` is a class variable, so `send_message` is a class
    method.
  * This class could be extended to also listen to the endpoint, instead of
    just blindly pushing messages out - after all, WebSockets allow for
    bidirectional data flow.
- The `RandomBernoulli` and `PeriodicCallback` make a pretty crude example, but
  you could write a class that transmits data in real-time to suit your use
  case. For example, you could watch a file for any modifications using
  [`watchdog`](https://pythonhosted.org/watchdog/), and dump the changes into
  the WebSocket.
- The [`websocket_ping_interval` and `websocket_ping_timeout` arguments to
  `tornado.Application`](https://www.tornadoweb.org/en/stable/web.html?highlight=websocket_ping#tornado.web.Application.settings)
  configure periodic pings of WebSocket connections, keeping connections alive
  and allowing dropped connections to be detected and closed.
- It's also worth noting that there's a
  [`tornado.websocket.WebSocketHandler.websocket_max_message_size`](https://www.tornadoweb.org/en/stable/websocket.html?highlight=websocket_max_message_size#tornado.websocket.WebSocketHandler)
  attribute. While this is set to a generous 10 MiB, it's important that the
  WebSocket messages don't exceed this limit!

<script src="https://gist.github.com/eigenfoo/22f46166fa6924d684d68ca06e08b055.js"></script>

## Client

- `WebSocketClient` is a class that:
  1. Can be `start`ed and `stop`ped to connect/disconnect to the WebSocket and
     start/stop listening to it in a separate thread
  2. Can process every message (`on_message`) it hears from the WebSocket: in
     this case it simply maintains [a count of the number of trials and
     successes](https://eigenfoo.xyz/bayesian-bandits/#stochastic-aka-stationary-bandits),
     but this processing could theoretically be anything. For example, you
     could do some further processing of the message and then dump that into a
     separate WebSocket for other apps (or even users!) to subscribe to.
- To connect to the WebSocket, we need to use a WebSocket library: thankfully
  Tornado has a built-in WebSocket functionality (`tornado.websocket`), but
  we're also free to use other libraries such as the creatively named
  [`websockets`](https://github.com/aaugustin/websockets) or
  [`websocket-client`](https://github.com/websocket-client/websocket-client).
- Note that we run `on_message` on the same thread as we run
  `connect_and_read`. This isn't a problem so long as `on_message` is fast
  enough, but a potentially wiser choice would be to offload `connect_and_read`
  to a separate thread by instantiating a
  [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor)
  and calling
  [`tornado.ioloop.IOLoop.run_in_executor`](https://www.tornadoweb.org/en/stable/ioloop.html#tornado.ioloop.IOLoop.run_in_executor),
  so as not to block the thread where the `on_message` processing happens.
- The `io_loop` instantiated in `main` (as well as in `server.py`) is
  important: it's how Tornado schedules tasks (a.k.a. _callbacks_) for delayed
  (a.k.a. _asynchronous_) execution. To add a callback, we simply call
  `io_loop.add_callback()`.
- The [`ping_interval` and `ping_timeout` arguments to
  `websocket_connect`](https://www.tornadoweb.org/en/stable/websocket.html?highlight=ping_#tornado.websocket.websocket_connect)
  configure periodic pings of the WebSocket connection, keeping connections
  alive and allowing dropped connections to be detected and closed.
- The `callback=self.maybe_retry_connection` is [run on a future
  `WebSocketClientConnection`](https://github.com/tornadoweb/tornado/blob/1db5b45918da8303d2c6958ee03dbbd5dc2709e9/tornado/websocket.py#L1654-L1655).
  `websocket_connect` doesn't actually establish the connection directly, but
  rather returns a future. Hence, we try to get the `future.result()` itself
  (i.e. the WebSocket client connection) — I don't actually do anything with
  the `self.connection`, but you could if you wanted. In the event of an
  exception while doing that, we assume there's a problem with the WebSocket
  connection and retry `connect_and_read` after 3 seconds. This all has the
  effect of recovering gracefully if the WebSocket is dropped or `server.py`
  experiences a brief outage for whatever reason (both of which are probably
  inevitable for long-running apps using WebSockets).

<script src="https://gist.github.com/eigenfoo/341f6c6c578d34120bccc4229e434377.js"></script>

## Why Tornado?

Tornado is a Python web framework, but unlike the more popular Python web
frameworks like [Flask](https://flask.palletsprojects.com/) or
[Django](https://www.djangoproject.com/), it has strong support for
[asynchronous networking and non-blocking
calls](https://www.tornadoweb.org/en/stable/guide/async.html#blocking) -
essentially, Tornado apps have one (single-threaded) event loop
(`tornado.ioloop.IOLoop`), which handles all requests asynchronously,
dispatching incoming requests to the relevant non-blocking function as the
request comes in. As far as I know, Tornado is the only Python web framework
that does this.

As an aside, Tornado seems to be [more popular in
finance](https://thehftguy.com/2020/10/27/my-experience-in-production-with-flask-bottle-tornado-and-twisted/),
where streaming real-time data (e.g. market data) is very common.

## Why WebSockets?

A sharper question might be, why WebSockets over HTTP requests to a REST
endpoint? After all, both theoretically allow a client to stream data in
real-time from a server.

[A lot can be said](https://stackoverflow.com/a/45464306) when comparing
WebSockets and RESTful services, but I think the main points are accurately
summarized by [Kumar Chandrakant on
Baeldung](https://www.baeldung.com/rest-vs-websockets#usage):

> [A] WebSocket is more suitable for cases where a push-based and real-time
> communication defines the requirement more appropriately. Additionally,
> WebSocket works well for scenarios where a message needs to be pushed to
> multiple clients simultaneously. These are the cases where client and server
> communication over RESTful services will find it difficult if not prohibitive.

Tangentially, there's one alternative that seems to be better than WebSockets
from a protocol standpoint, but unfortunately doesn't seem to have support from
many Python web frameworks, and that is [Server-Sent Events (a.k.a.
SSE)](https://www.smashingmagazine.com/2018/02/sse-websockets-data-flow-http2/):
it seems to be a cleaner protocol for unidirectional data flow, which is really
all that we need.

Additionally, [Armin
Ronacher](https://lucumr.pocoo.org/2012/9/24/websockets-101/) has a much
starker view of WebSockets, seeing no value in using WebSockets over TCP/IP
sockets for this application:

> Websockets make you sad. [...] Websockets are complex, way more complex than I
> anticipated. I can understand that they work that way but I definitely don't
> see a value in using websockets instead of regular TCP connections if all you
> want is to exchange data between different endpoints and neither is a browser. 

My thought after reading these criticisms is that perhaps WebSockets aren't the
ideal technology for handling streaming data (from a maintainability or
architectural point of view), but that doesn't mean that they aren't good
scalable technologies when they do work.

---

[^1]: There is [technically a difference](https://sqlstream.com/real-time-vs-streaming-a-short-explanation/) between "real-time" and "streaming": "real-time" refers to data that comes in as it is created, whereas "streaming" refers to a system that processes data continuously. You stream your TV show from Netflix, but since the show was created long before you watched it, you aren't viewing it in real-time.
