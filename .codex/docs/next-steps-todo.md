# TODOs

- add deck info to obs
  - either an enum for which deck (cannot use custom decks)
  - process set of all deck cards (best but will blow up obs space more)

- add obs for player last action
  - not sure to also add last opponent action as well? Ablate on that

- mess with reward shaping some more, I think right now its impact is really small
  - also add time based rewards if thats not already a thing to enforce faster actioning (i think it is)
    - might need to adjust the time degradation
