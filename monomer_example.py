"""
PyLabRobot + Opentrons OT-2: basic liquid handling example for Monomer hackathon participants.

Demonstrates: setup deck, pick up tips, aspirate/dispense (with mixing), return tips.
Transfer flow: 24-well plate → 96-well deep → 96-well flat. Uses both tip types
(P300 left/channel 0, P1000 right/channel 1). Mixing is done by passing
mix=[Mix(volume_µL, repetitions, flow_rate_µL/s)] to aspirate() and/or dispense().
Pipetting height is set via liquid_height (mm above well bottom).

Deck layout (slots):
  Slot 1: 24-well deepwell plate, 10 mL (source)
  Slot 2: 96-well deepwell plate, 2.2 mL (intermediate)
  Slot 3: 96-well flat bottom plate, 360 µL (destination)
  Slot 6: 200 µL filter tip rack (P300 / channel 0; tip max 200 µL)
  Slot 9: 1000 µL filter tip rack (P1000 / channel 1; tip max 1000 µL)

IMPORTANT: On the OT-2 dual pipette, pass use_channels=[0] for left (P300) or
use_channels=[1] for right (P1000). Tip rack and pipette must match.

Error handling:
  On any exception or Ctrl+C, the script discards any mounted tips to waste, homes
  the robot, then re-raises the error. This keeps the robot in a safe state when something
  goes wrong.
"""
import asyncio
import logging
import os
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import OpentronsOT2Backend, OpentronsOT2Simulator
from pylabrobot.liquid_handling.standard import Mix
from pylabrobot.resources import (
    OTDeck,
    Cor_96_wellplate_360ul_Fb,
    Cor_Axy_24_wellplate_10mL_Vb,
    NEST_96_wellplate_2200uL_Ub,
)
from pylabrobot.resources.opentrons.load import load_ot_tip_rack

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Deck setup
# -----------------------------------------------------------------------------
OT2_HOST = os.environ.get("OT2_HOST")
if not OT2_HOST:
    logger.error("OT2_HOST environment variable is not set")
    raise ValueError(
        "OT2_HOST environment variable is not set. "
        "Set it to your robot's IP, or use 'simulate' for offline testing."
    )
deck = OTDeck()
if OT2_HOST == "simulate":
    backend = OpentronsOT2Simulator(
        left_pipette_name="p300_single_gen2",
        right_pipette_name="p1000_single_gen2",
    )
else:
    backend = OpentronsOT2Backend(host=OT2_HOST)
lh = LiquidHandler(backend=backend, deck=deck)

tip_200 = load_ot_tip_rack("opentrons_96_filtertiprack_200ul", "200uL")
tip_1000 = load_ot_tip_rack("opentrons_96_filtertiprack_1000ul", "1000uL")
deck.assign_child_at_slot(tip_200, 6)
deck.assign_child_at_slot(tip_1000, 9)

plate_24_deep = Cor_Axy_24_wellplate_10mL_Vb(name="plate_24_deep")
plate_96_deep = NEST_96_wellplate_2200uL_Ub(name="plate_96_deep")
plate_96_flat = Cor_96_wellplate_360ul_Fb(name="plate_96_flat")
deck.assign_child_at_slot(plate_24_deep, 1)
deck.assign_child_at_slot(plate_96_deep, 2)
deck.assign_child_at_slot(plate_96_flat, 3)

async def cleanup():
    """Discard any mounted tips to waste and home. Used after error or cancel."""
    logger.info("Cleanup: discarding tips and homing")
    try:
        await lh.discard_tips()
    except Exception as e:
        logger.warning("Could not discard tips: %s — eject tips manually if needed", e)
    try:
        await backend.home()
    except Exception as e:
        logger.warning("Home during cleanup failed: %s", e)


async def run_transfers():
    """Run the full pipetting protocol. Raises on error; caller handles cleanup."""
    logger.info("Starting protocol: setup (homing)")
    await lh.setup(skip_home=False)

    # P1000: 24-well → 96-deepwell (with mix at source)
    logger.info("Transfer 1 (P1000): 24-well A1 → 96-deep A1, 200 µL with mix")
    await lh.pick_up_tips(tip_1000["A1"], use_channels=[1])
    await lh.aspirate(
        plate_24_deep["A1"],
        vols=[200],
        mix=[Mix(volume=200, repetitions=3, flow_rate=400)],
        liquid_height=[1.0],
        use_channels=[1],
    )
    await lh.dispense(
        plate_96_deep["A1"],
        vols=[200],
        liquid_height=[1.0],
        use_channels=[1],
    )
    await lh.return_tips()

    # P300: 96-deepwell → 96-flat (with mix at source; 200 µL max tip volume)
    logger.info("Transfer 2 (P300): 96-deep A1 → 96-flat A1, 100 µL with mix")
    await lh.pick_up_tips(tip_200["A1"], use_channels=[0])
    await lh.aspirate(
        plate_96_deep["A1"],
        vols=[100],
        mix=[Mix(volume=100, repetitions=3, flow_rate=400)],
        liquid_height=[1.0],
        use_channels=[0],
    )
    await lh.dispense(
        plate_96_flat["A1"],
        vols=[100],
        liquid_height=[1.0],
        use_channels=[0],
    )
    await lh.return_tips()

    logger.info("Protocol complete: homing")
    await backend.home()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        await run_transfers()
        logger.info("Protocol finished successfully")
    except BaseException:
        logger.warning("Protocol interrupted or failed; running cleanup")
        await cleanup()
        raise


if __name__ == "__main__":
    asyncio.run(main())
