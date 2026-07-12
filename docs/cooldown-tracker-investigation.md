# Enemy Cooldown Tracker Investigation

## Decision

Do not implement or distribute the proposed in-game enemy ultimate cooldown timer
without written Riot approval.

Riot's current Game Integrity policy explicitly prohibits products that provide
information not present in the game client for a competitive edge and gives
"automatically or manually allowing tracking enemy ultimate cooldowns" as its
example. A user-clicked approximate timer is therefore still inside the stated
prohibition. Enemy summoner-spell timers carry the same material policy risk.

- Riot League of Legends developer policy:
  https://developer.riotgames.com/docs/lol#game-integrity
- Riot Developer Relations policy copy:
  https://support-developer.riotgames.com/hc/en-us/articles/22698698001939-League-of-Legends

## Verified technical findings

- The Live Client Data API remains documented and available at
  `https://127.0.0.1:2999/liveclientdata` during an active match.
- `playerlist` provides enemy champion, level, and summoner-spell metadata. It does
  not provide enemy cast events or enemy cooldown state.
- Data Dragon remains available for patch-static champion/summoner metadata and
  icons. It is manually published, may lag a game patch, and should be cached by
  the realm's reported version.
- The legacy Ult Tracker's Fandom item-data request now returns HTTP 403 and must not
  be retained.
- Current champion ultimate data is not uniformly three ranks: the data includes
  one-, three-, four-, and six-rank abilities, plus zero/unsupported cooldown data.
  A generic levels 6/11/16 assumption is not safe without explicit exceptions.
- Smite's static cooldown is not its useful charge-recharge behavior and must be
  modeled as unsupported rather than shown as a normal timer.

Official references:

- Live Client API and sample response:
  https://developer.riotgames.com/docs/lol#game-client-api_live-client-data-api
  https://static.developer.riotgames.com/docs/lol/liveclientdata_sample.json
- Data Dragon:
  https://developer.riotgames.com/docs/lol#data-dragon
  https://ddragon.leagueoflegends.com/api/versions.json

## Legacy repository assessment

The v1.0.0 release source matches the locally recovered source, but the implementation
should not be merged. It performs network work during module import, lacks consistent
timeouts/schema validation, uses one drifting sleep thread per timer, mutates shared
state without synchronization, has no match lifecycle or orderly shutdown, can
misassociate asynchronously loaded icons, and has no tests or packaging discipline.

Only the general Data Dragon icon/static-data concept is worth retaining. The minimap
tracker's typed adapters, cached data client, injected monotonic clock, Qt event loop,
and tested lifecycle are the appropriate foundation for any approved future work.

## Compliant alternatives

- A pre-game static cooldown reference with no game-session timer state.
- A replay/practice-tool review screen that is not used to provide live competitive
  information.
- Self or allied-team cooldown information explicitly approved by Riot.

## Design if Riot grants written approval

- Keep the controls in a separate interactive Qt panel; the minimap overlay must
  remain click-through.
- Normalize stable enemy IDs, level, and two summoner spell IDs from Live Client data.
- Cache patch-scoped champion-full and summoner Data Dragon payloads and icons.
- Use a single pure cooldown engine with monotonic `ready_at` deadlines and one Qt
  refresh timer. Never start a worker thread per icon.
- Snapshot level/rank and base cooldown when clicked; later level changes must not
  alter an already-running timer.
- Left-click starts/restarts, right-click cancels, and unsupported/charge-based
  abilities remain visibly disabled with an explanation.
- Reset every timer on confirmed match end or roster/session replacement.
- Test schema failures, offline cache fallback, nonstandard ultimate ranks,
  monotonic boundaries, session reset, and Qt mouse behavior with fixtures.
