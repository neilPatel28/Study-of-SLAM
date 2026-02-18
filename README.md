# SLAM from Scratch in Python

A step-by-step implementation of Simultaneous Localization and Mapping (SLAM) for understanding how robots navigate and map unknown environments. Each module builds toward a full SLAM system using particle filters.

## Overview

This project breaks SLAM into three digestible pieces, implemented one at a time.

The first module is a particle filter for localization given a known map. It uses a noisy motion model to propagate particles and a sensor likelihood model to update particle weights, then resamples to focus particles on high-probability regions.

The second module handles mapping given a known true pose. It builds an occupancy grid map from sensor readings using log-odds updates, accounting for noisy motion and sensor models.

The third module, currently in progress, combines both — estimating the robot's pose and the map simultaneously with no prior knowledge of either. This will use a Rao-Blackwellized Particle Filter (FastSLAM) approach.

## Status

Particle filter localization is complete. Occupancy grid mapping is complete. Full SLAM is in progress in python. Future work will be to relpticate a simulation based SLAM paper. 

## References

This project is an extension of the Mobile Robotics course by Dr. David Rosen, implementing the taught mathematical concepts from the course material into real coding examples to deepen understanding of robotics.
