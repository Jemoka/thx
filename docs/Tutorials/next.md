# What's next?

This spiel covered a _very tiny slice_ of theseus's features. However, the
remainder of our story today will be a choose-your-own-adventure. Take a look
at the how-to guide sidebar and decide what you want to poke around in :)

I'll give you some ideas here as things that I'm particularly excited about
in theseus, in no particular order... here goes:

## Evaluations

Almost as easy as adding datasets, theseus allows you to add evaluations and
run them inline during training, without having to worry about such rudimentary
things as KV caches (:gasp!), rollout logistics (:gasp!), or perplexity
calculations (:gasp!).

Pick your strategy and wire it into a trainer with the
[Evaluation System guide](../How-to/evaluation.md), or
[add your own evaluation](../How-to/adding/components/evaluation.md).

## Draw pretty pictures during and after training

Yk the time-travel debugging API I told you about in
[inspecting runs](inspecting-runs.md)? You can actually formalize those little
experiments into full-fledged jobs and **attach them to historical nodes**,
right alongside the records from training.

Learn about the `@analysis` decorator, checkpoint artifacts, and plots during
training in [Analysis System](../How-to/analysis.md). When you're ready to write
one, head to [Adding an analysis job](../How-to/adding/components/analysis-job.md).

## The configuration system

Ask for any field, any time... once your component declares where it belongs.
Your model, dataset, optimizer, and other ingredients bring their configuration
fields along with them. There's no giant central config class to keep feeding.

Read about [how the configuration system fits together](../Design/config.md),
or [add an experiment](../How-to/adding/components/experiment.md) that puts it to work.

## JuiceFS integration

It's a bit annoying to have to copy root directories around. How about we let
our machines look at the same files instead?

The [JuiceFS Integration guide](../How-to/remote-configuration/juicefs.md) covers mounting a shared
root, using it from a notebook, and figuring out why the filesystem is giving
you the cold shoulder.


## Even more boxes to dispatch to

I covered a very minimal example of how to dispatch a theseus job remotely.
There's like three platforms we support—SSH, SLURM, and Volcano K8s—and a billion
configuration options. Meet them in
[Remote Configuration](../How-to/remote-configuration/providers.md).

If you're picking hardware, here's the [big ol' chip table](../Design/chips.md).
If you're deciding how those chips should share the work, here's the
[sharding design](../Design/sharding.md).

## Make more things your own

Still itching to replace another part? [Adding Things](../How-to/adding/index.md)
points you to [models](../How-to/adding/components/model.md),
[experiments](../How-to/adding/components/experiment.md),
[datasets](../How-to/adding/components/dataset.md),
[evaluations](../How-to/adding/components/evaluation.md), and
[analysis jobs](../How-to/adding/components/analysis-job.md).
