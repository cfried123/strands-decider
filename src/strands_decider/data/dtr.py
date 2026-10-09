"""DTR Part I (Passenger Movement) recipes: travel-entitlement decisions as choice,
noul and score questions.

Each task is a family of scenario templates grounded in one provision of the Defense
Transportation Regulation Part I. A scenario is drawn from a small table of decisive
facts, so its gold label follows from the cited provision rather than from a model.
The state carries the scenario, the citation and the verbatim excerpt, so the model is
trained to read the rule it is applying.

Ambiguous noul scenarios (the decisive fact is missing) are emitted twice, once per
label at weight 0.5: the expected cross-entropy is minimised at p(true) = 0.5, which
trains the calibrated abstention the review queue relies on.

`dtr_bench` builds the DTR-Bench gold set from disjoint traveler/route vocabularies so
no bench scenario string appears in the training corpus.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from .format import Example

SOURCE = "https://www.business.ustranscom.mil/dtr/dtrp1.cfm"

# Verbatim excerpts (whitespace-normalised) from DTR Part I chapters 102-107.
EXCERPTS: dict[str, tuple[str, str]] = {
    "102.A": (
        "DTR Part I, Ch. 102, para A.2-A.4 (Travel Authority / Class of Service)",
        "It is government policy that all travelers must be provided economy or coach airline "
        "accommodations for all official travel unless proper justification and documentation are "
        "provided (ordinarily before travel) and substantiated to justify premium class "
        "accommodations. Economy or coach class includes economy plus or coach elite seating on "
        "airlines. ... The authorizing official must authorize or approve use of this seating "
        "service for it to be reimbursable by the Government. Premium economy is lower than both "
        "first class and business class, but higher than economy plus or coach elite. The "
        "authorizing official must authorize or approve use of this seating service for it to be "
        "reimbursable by the Government. Premium class travel is only permitted on an exception "
        "basis, when fully justified and approved by the premium class approval authority ... "
        "Blanket authorization and justification for use of premium class travel is prohibited. "
        "In the event premium class tickets are issued without proper approval documentation, the "
        "traveler is responsible for the excess cost. Travelers are authorized to upgrade to "
        "premium class at their personal expenses or with frequent traveler benefits.",
    ),
    "102.B": (
        "DTR Part I, Ch. 102, para B (Dual Commitment Transportation)",
        "Travelers or government entities are prohibited from making duplicate bookings on any "
        "mode of transportation or working to have two types of transportation (military and "
        "commercial) simultaneously. Individual or group commercial air, rail, or bus "
        "transportation will not be requested or scheduled unless all actions to obtain military "
        "transportation have been terminated.",
    ),
    "102.C": (
        "DTR Part I, Ch. 102, para C.1 (Routing); Ch. 104, para A.1; Ch. 105, para A.2",
        "Commercial transportation must be used for DoD passengers in the Continental United "
        "States (CONUS) except when military transportation is essential to meet training or "
        "mission requirements, or when opportune capability is generated as a result of training "
        "or logistics support requirements. When travel orders do not direct a specific mode of "
        "transportation, or the directed mode is not available, the TO arranging transportation "
        "must route passengers using the best value that meets mission requirements. [104.A.1] "
        "Because of the passenger count and baggage requirements of group movements, bus "
        "transportation is often the most economical and efficient method of moving Department "
        "of Defense (DoD) personnel. [105.A.2] Rail will be provided only when it meets mission "
        "requirements and is the most cost-effective mode, taking into consideration best value, "
        "such as travel time, per diem, and miscellaneous expenses.",
    ),
    "102.J": (
        "DTR Part I, Ch. 102, para J (Circuitous Travel)",
        "Circuitous travel is defined as travel by a route other than the one that would normally "
        "be prescribed by a TO between the places listed in authorization travel orders. ... The "
        "designated official approves or disapproves requests for circuitous travel. ... The "
        "traveler assumes responsibility for all costs exceeding those of the normally prescribed "
        "route.",
    ),
    "102.K": (
        "DTR Part I, Ch. 102, para K (TMC)",
        "When a TMC is available, it is DoD policy that TMC use by travelers is mandatory IAW JTR "
        "par. 010201, \u201cKey Participants\u201d.",
    ),
    "103.E": (
        "DTR Part I, Ch. 103, para E.1-E.2 (GSA City Pair Program)",
        "GSA City Pair fares cannot be used to and/or from an unofficial point (such as a leave "
        "point) unless the Government is paying for the leave travel to (and/or from) the leave "
        "point. ... Exception to the Use of GSA CPP Contract Carriers: One or more of the "
        "following travel conditions, which must be certified on the travel order, travel "
        "voucher, or other document ... (1) Space or a scheduled flight ... is not available in "
        "time to accomplish the purpose of travel, or use of contract service would require the "
        "traveler to incur unnecessary overnight lodging costs ... (2) The contract carrier\u2019s "
        "flight schedule is inconsistent with explicit policies ... to schedule travel during "
        "normal working hours. (3) A non-contract U.S. flag carrier offers a lower fare available "
        "to the general public, the use of which will result in a lower total trip cost ... This "
        "exception does not apply if the contract carrier(s) offers the same or lower fare and "
        "has seats available at that fare, or if the lower fare offered by a non-contract carrier "
        "is restricted to the Government and military travelers on official business ... "
        "Contractor personnel are not authorized to use Government discount fares (including GSA "
        "City Pair fares) ... DoD personnel traveling on official business may not purchase "
        "commercial airline tickets at military furlough fares ... A movement of less than 10 "
        "members traveling as a group ... is a mandatory user of the GSA CPP. A group movement of "
        "10 or more members ... is a non-mandatory user of the GSA CPP.",
    ),
    "103.F": (
        "DTR Part I, Ch. 103, para F (Use of Foreign Flag Carriers)",
        "IASA Category 1 Countries. Unless otherwise restricted by the DoD, air carriers from "
        "non-rated or Category 1 countries are approved for use for official travel on an "
        "individually ticketed, scheduled-service basis only and will not apply to DoD charter "
        "operations or group travel. IASA Category 2 Countries. Air carriers from Category 2 "
        "countries may not be used by DoD personnel for official business travel, except: (1) On "
        "those legs of a carrier\u2019s route that starts or ends in the United States. (2) In "
        "extenuating circumstances ... and the travel is mission-essential, the Geographic "
        "Combatant Commander (GCC) or Deputy GCC may authorize the use of the foreign carrier with "
        "a written waiver ... travel under a ticket issued by a U.S. certified air carrier which "
        "uses space on the aircraft of a foreign flag air carrier under a code-share arrangement "
        "is considered to be transportation provided by U.S. air carriers ... In instances where a "
        "foreign air carrier engages in code-share arrangements with another foreign air carrier, "
        "both air carriers must be from either a FAA-non-rated and/or FAA rated Category 1 "
        "country.",
    ),
    "103.N": (
        "DTR Part I, Ch. 103, para N (Defense Contractor Personnel)",
        "Defense contractor personnel may be provided transportation on DoD airlift when the "
        "Contracting Officer (CO/KO) issues a signed Letter of Authorization (LOA). All LOAs must "
        "contain a statement that commercial transportation is not available, obtainable, or "
        "capable of meeting the transportation requirement. The LOA will also identify the period "
        "of travel, travel origin and destination, prime contract number, sub-contract number, "
        "and emergency contact phone number.",
    ),
    "106.B": (
        "DTR Part I, Ch. 106, para B (Rental Services Policy)",
        "TOs, TMCs, and travelers must select the lowest-cost rental service that meets the "
        "traveler\u2019s official requirements. It is mandatory to obtain rental vehicle "
        "reservations through the TMC, when available. ... Rental car use is limited to official "
        "purposes, including transportation to and from: Duty Sites, Lodgings, Dining facilities, "
        "Drugstores, Barber shops, Places of worship, Cleaning establishments, Similar places "
        "required for the traveler\u2019s subsistence, health, or comfort.",
    ),
    "107.A": (
        "DTR Part I, Ch. 107, para A (Commercial Ship Transportation)",
        "Travel by ship to or from the Continental United States (CONUS) and Outside CONUS "
        "(OCONUS) is not usually an authorized mode for transoceanic transportation. ... The use "
        "of commercial ships must be authorized or approved in extenuating or emergency situations "
        "by the order-issuing authority. ... a. The travel may be completed only by use of ship. "
        "b. The travel can be performed more economically or efficiently by ship. c. Cognizant "
        "command medical authority certifies the traveler as medically unable to use air "
        "transportation and ship travel is the only reasonable alternative. When transportation "
        "by ship is authorized/approved, a United States (U.S.) registry ship must be used. If a "
        "U.S. registry ship cannot provide transportation service ... transportation may be "
        "obtained aboard a foreign registry ship. Required documentation as to the "
        "non-availability of a U.S. registry ship must be provided to the traveler.",
    ),
}

# Disjoint vocabularies: training draws from the first half, DTR-Bench from the second.
TRAVELERS = (
    [
        "an Army captain (O-3)",
        "a GS-12 DoD civilian employee",
        "a Navy petty officer first class (E-6)",
        "a Marine Corps lieutenant colonel (O-5)",
        "an Air Force senior airman (E-4)",
        "a GS-14 DoD civilian analyst",
        "a Space Force first lieutenant (O-2)",
        "a Coast Guard chief warrant officer (W-3)",
    ],
    [
        "an Army sergeant first class (E-7)",
        "a GS-13 DoD civilian engineer",
        "a Navy commander (O-5)",
        "a Marine Corps staff sergeant (E-6)",
        "an Air Force major (O-4)",
        "a GS-11 DoD civilian logistician",
        "a Space Force master sergeant (E-7)",
    ],
)
CONUS_ROUTES = (
    [
        ("Washington, DC", "San Diego, CA"),
        ("Fort Liberty, NC", "El Paso, TX"),
        ("Norfolk, VA", "Seattle, WA"),
        ("Colorado Springs, CO", "Huntsville, AL"),
        ("Tampa, FL", "Omaha, NE"),
        ("Dayton, OH", "Sacramento, CA"),
    ],
    [
        ("Boston, MA", "Albuquerque, NM"),
        ("San Antonio, TX", "Philadelphia, PA"),
        ("Jacksonville, FL", "Salt Lake City, UT"),
        ("Columbus, GA", "Spokane, WA"),
    ],
)
OCONUS_ROUTES = (
    [
        ("Norfolk, VA", "Naples, Italy"),
        ("Honolulu, HI", "Manila, Philippines"),
        ("Ramstein, Germany", "Nairobi, Kenya"),
        ("Atlanta, GA", "Seoul, South Korea"),
    ],
    [
        ("Dallas, TX", "Tokyo, Japan"),
        ("Stuttgart, Germany", "Accra, Ghana"),
        ("Chicago, IL", "Warsaw, Poland"),
        ("Guam", "Jakarta, Indonesia"),
    ],
)
PURPOSES = (
    ["a TDY training course", "a program review", "a site survey", "a planning conference"],
    [
        "an inspection visit",
        "a contract kickoff meeting",
        "an exercise planning event",
        "a staff assistance visit",
    ],
)


@dataclass
class Ctx:
    """Vocabulary half (0 train, 1 bench) and the RNG that fills scenario slots."""

    half: int
    rng: random.Random
    turn: int = 0

    def k(self, n: int) -> int:
        """Case index: random for training, round-robin for the bench so every case
        template is represented."""
        self.turn += 1
        return self.rng.randrange(n) if self.half == 0 else (self.turn - 1) % n

    def pick(self, table: tuple[list, list]):
        return self.rng.choice(table[self.half])

    def who(self) -> str:
        return self.pick(TRAVELERS)

    def conus(self) -> str:
        a, b = self.pick(CONUS_ROUTES)
        return f"from {a} to {b}"

    def oconus(self) -> str:
        a, b = self.pick(OCONUS_ROUTES)
        return f"from {a} to {b}"

    def why(self) -> str:
        return self.pick(PURPOSES)


@dataclass
class Case:
    """One generated scenario: `label` is an option name, or None when ambiguous."""

    scenario: str
    label: str | None
    section: str


def _state(case: Case, with_excerpt: bool = True) -> dict:
    cite, text = EXCERPTS[case.section]
    state = {"scenario": case.scenario, "dtr_citation": cite}
    if with_excerpt:
        state["dtr_excerpt"] = text
    return state


# ---------------------------------------------------------------- choice: class of service
CLASS_OPTIONS = [
    ["economy_coach", "Economy or coach class at Government expense"],
    ["economy_plus", "Economy plus / coach elite seating at Government expense"],
    ["premium_economy", "Premium economy at Government expense"],
    ["premium_class", "Business or first class (premium class) at Government expense"],
    [
        "personal_upgrade",
        "Premium class only as an upgrade at the traveler's personal expense "
        "or with frequent traveler benefits",
    ],
]
CLASS_INSTR = [
    "Which class of accommodation is authorized for this trip under the DTR?",
    "Under DTR Part I Chapter 102, what accommodation class may this traveler use, and at "
    "whose expense?",
    "Determine the authorized class of service for this official travel.",
]


def gen_class_of_service(c: Ctx) -> Case:
    w, r, y = c.who(), c.rng.choice([c.conus(), c.oconus()]), c.why()
    base = f"{w[0].upper() + w[1:]} is flying {r} for {y}. "
    k = c.k(9)
    if k == 0:
        return Case(
            base + "The TMC booked a standard economy seat; no other class was requested.",
            "economy_coach",
            "102.A",
        )
    if k == 1:
        return Case(
            base + "The traveler asked for an economy plus seat with extra legroom, and "
            "the authorizing official approved it in the travel authorization.",
            "economy_plus",
            "102.A",
        )
    if k == 2:
        return Case(
            base + "The traveler wants an economy plus seat for extra legroom, but the "
            "authorizing official has not authorized or approved it.",
            "economy_coach",
            "102.A",
        )
    if k == 3:
        return Case(
            base + "The authorizing official approved premium economy in the travel "
            "authorization because of the flight length.",
            "premium_economy",
            "102.A",
        )
    if k == 4:
        return Case(
            base + "The traveler requested premium economy; no approval from the "
            "authorizing official is on file.",
            "economy_coach",
            "102.A",
        )
    if k == 5:
        return Case(
            base + "A documented medical disability justifies business class, and the "
            "premium class approval authority's approval is part of the travel "
            "authorization.",
            "premium_class",
            "102.A",
        )
    if k == 6:
        return Case(
            base + "The office relies on a blanket authorization that permits business "
            "class for all of its senior staff; no trip-specific justification exists.",
            "economy_coach",
            "102.A",
        )
    if k == 7:
        return Case(
            base + "The traveler holds an economy ticket and wants to use frequent "
            "traveler miles to move up to business class.",
            "personal_upgrade",
            "102.A",
        )
    return Case(
        base + "The traveler asks the TMC for a first class seat; there is no premium "
        "class approval in the travel authorization.",
        "economy_coach",
        "102.A",
    )


# ---------------------------------------------------------------- choice: mode of travel
MODE_OPTIONS = [
    ["commercial_air", "Commercial air (scheduled airline service)"],
    ["military_airlift", "Military (DoD) airlift"],
    ["rail", "Rail (e.g., Amtrak)"],
    ["charter_bus", "Charter bus from a DoD-approved carrier"],
]
MODE_INSTR = [
    "Which mode of transportation should the transportation officer arrange?",
    "Under DTR Part I routing policy, what is the authorized mode for this movement?",
    "Select the mode of travel the DTR directs for this scenario.",
]


def gen_travel_mode(c: Ctx) -> Case:
    w, r, y = c.who(), c.conus(), c.why()
    k = c.k(6)
    if k == 0:
        return Case(
            f"{w[0].upper() + w[1:]} is traveling alone within CONUS {r} for {y}. Orders do "
            "not direct a mode; there is no training or mission requirement for military "
            "transportation and no rail option is cost-effective.",
            "commercial_air",
            "102.C",
        )
    if k == 1:
        return Case(
            f"{w[0].upper() + w[1:]} must move within CONUS {r} with a unit for {y}, and "
            "military transportation is essential to meet the unit's training "
            "requirement.",
            "military_airlift",
            "102.C",
        )
    if k == 2:
        return Case(
            f"{w[0].upper() + w[1:]} is traveling within CONUS {r}. A C-17 already flying the "
            "route for a logistics support mission has opportune seats available.",
            "military_airlift",
            "102.C",
        )
    if k == 3:
        a, b = c.rng.choice(
            [
                ("Washington, DC", "New York, NY"),
                ("Philadelphia, PA", "Washington, DC"),
                ("Boston, MA", "Providence, RI"),
            ]
        )
        return Case(
            f"{w[0].upper() + w[1:]} is traveling from {a} to {b} for {c.why()}. Rail meets "
            "the mission requirement and, counting travel time and per diem, is the most "
            "cost-effective mode.",
            "rail",
            "102.C",
        )
    if k == 4:
        n = c.rng.choice([28, 35, 44, 52])
        return Case(
            f"A group of {n} personnel with unit baggage must move about 180 miles "
            f"between installations for {y}. Moving them together by road is the most "
            "economical option that meets the mission.",
            "charter_bus",
            "102.C",
        )
    return Case(
        f"{w[0].upper() + w[1:]} prefers to drive a government van {r} for {y}, but orders "
        "direct no mode, the distance is over 1,500 miles, and best value that meets the "
        "mission is a scheduled flight.",
        "commercial_air",
        "102.C",
    )


# ---------------------------------------------------------------- noul families
NOUL_DESC = ("The DTR does not authorize it", "The DTR authorizes it")


def gen_fare_eligibility(c: Ctx) -> Case:
    w, r = c.who(), c.conus()
    k = c.k(8)
    if k == 0:
        return Case(
            f"{w[0].upper() + w[1:]} on official TDY {r} is booked on a GSA City Pair "
            "contract fare for both official legs.",
            "true",
            "103.E",
        )
    if k == 1:
        return Case(
            f"{w[0].upper() + w[1:]} is taking personal leave in Orlando after TDY {r}. The "
            "Government is not paying for the leave travel, and the traveler wants the "
            "City Pair fare on the leave leg.",
            "false",
            "103.E",
        )
    if k == 2:
        return Case(
            f"A defense contractor employee supporting a program office wants to buy a "
            f"GSA City Pair fare {r} for a design review.",
            "false",
            "103.E",
        )
    if k == 3:
        return Case(
            f"{w[0].upper() + w[1:]} on official business {r} found a military furlough fare "
            "(fare basis BM) cheaper than the contract fare and wants to use it.",
            "false",
            "103.E",
        )
    if k == 4:
        n = c.rng.choice([4, 6, 8, 9])
        return Case(
            f"A group of {n} members traveling together {r} wants a non-contract carrier "
            "because the group prefers its loyalty program.",
            "false",
            "103.E",
        )
    if k == 5:
        n = c.rng.choice([12, 15, 22])
        return Case(
            f"A group of {n} members is moving together {r}; the TO chose a non-contract "
            "carrier that best meets the travel requirement.",
            "true",
            "103.E",
        )
    if k == 6:
        return Case(
            f"{w[0].upper() + w[1:]} is adding a personal stop in Denver to an official trip "
            f"{r}; the file does not say whether the Government is paying for travel to "
            "the leave point.",
            None,
            "103.E",
        )
    return Case(
        f"A traveler {r} wants a GSA City Pair fare; the request does not say whether "
        "the traveler is a DoD civilian employee or contractor personnel.",
        None,
        "103.E",
    )


def gen_cpp_exception(c: Ctx) -> Case:
    w, r = c.who(), c.conus()
    base = f"{w[0].upper() + w[1:]} is flying {r} on a GSA City Pair contract route. "
    k = c.k(8)
    if k == 0:
        return Case(
            base + "No contract flight has seats in time for the mission, and the AO "
            "certified this on the travel order before booking a non-contract carrier.",
            "true",
            "103.E",
        )
    if k == 1:
        return Case(
            base + "The only contract flight would force an extra unnecessary hotel "
            "night; the AO certified this on the order and a non-contract flight was "
            "booked.",
            "true",
            "103.E",
        )
    if k == 2:
        return Case(
            base + "A non-contract U.S. flag carrier offers a lower fare available to the "
            "general public that lowers total trip cost, certified on the order.",
            "true",
            "103.E",
        )
    if k == 3:
        return Case(
            base + "The traveler booked a non-contract carrier to earn miles in a "
            "preferred frequent-flyer program.",
            "false",
            "103.E",
        )
    if k == 4:
        return Case(
            base + "A non-contract carrier's YDG fare is cheaper, but that fare is "
            "restricted to Government travelers and only purchasable with the "
            "Government charge card.",
            "false",
            "103.E",
        )
    if k == 5:
        return Case(
            base + "A non-contract carrier is cheaper, but the contract carrier offers "
            "the same fare with seats available.",
            "false",
            "103.E",
        )
    if k == 6:
        return Case(
            base + "A non-contract carrier shows a lower fare, but the request does not "
            "say whether that fare is available to the general public or restricted to "
            "Government travelers.",
            None,
            "103.E",
        )
    return Case(
        base + "The traveler says the contract flight 'won't work' and booked another "
        "airline; no reason or certification has been provided yet.",
        None,
        "103.E",
    )


def gen_foreign_carrier(c: Ctx) -> Case:
    w, r = c.who(), c.oconus()
    base = f"{w[0].upper() + w[1:]} is traveling {r} on official business. "
    k = c.k(9)
    if k == 0:
        return Case(
            base + "One leg is on an individually ticketed scheduled flight of an air "
            "carrier from an IASA Category 1 country.",
            "true",
            "103.F",
        )
    if k == 1:
        return Case(
            base + "The unit wants to charter an aircraft from a Category 1 country "
            "carrier for a group of 40 personnel.",
            "false",
            "103.F",
        )
    if k == 2:
        return Case(
            base + "An intermediate leg, neither starting nor ending in the United "
            "States, is on a carrier from an IASA Category 2 country; no waiver exists.",
            "false",
            "103.F",
        )
    if k == 3:
        return Case(
            base + "The leg departing the United States is operated by a carrier from an "
            "IASA Category 2 country.",
            "true",
            "103.F",
        )
    if k == 4:
        return Case(
            base + "Travel is mission-essential, no alternative exists, and the "
            "Geographic Combatant Commander signed a written waiver to use a Category 2 "
            "carrier on an intra-theater leg.",
            "true",
            "103.F",
        )
    if k == 5:
        return Case(
            base + "A U.S. airline issued the entire ticket under its code; the flight is "
            "operated by a foreign partner from a Category 1 country under a code-share.",
            "true",
            "103.F",
        )
    if k == 6:
        return Case(
            base + "A foreign carrier from a Category 1 country sold the ticket under a "
            "code-share, but the operating carrier is from a Category 2 country.",
            "false",
            "103.F",
        )
    if k == 7:
        return Case(
            base + "An intermediate leg is on a foreign carrier whose country's IASA "
            "category has not been determined.",
            None,
            "103.F",
        )
    return Case(
        base + "A staff officer says the GCC 'is fine with' a Category 2 carrier on an "
        "intra-theater leg; whether a written waiver exists is unknown.",
        None,
        "103.F",
    )


def gen_contractor_airlift(c: Ctx) -> Case:
    r = c.oconus()
    base = f"A defense contractor technician needs to travel {r} to support a fielded system. "
    k = c.k(6)
    if k == 0:
        return Case(
            base + "The contracting officer issued a signed LOA stating commercial "
            "transportation cannot meet the requirement, with travel period, origin, "
            "destination, contract numbers and emergency contact.",
            "true",
            "103.N",
        )
    if k == 1:
        return Case(
            base + "The program manager emailed the terminal asking that the technician "
            "be manifested; no LOA was issued.",
            "false",
            "103.N",
        )
    if k == 2:
        return Case(
            base + "A draft LOA exists but the contracting officer has not signed it.",
            "false",
            "103.N",
        )
    if k == 3:
        return Case(
            base + "The contracting officer signed an LOA, but it omits any statement "
            "that commercial transportation is unavailable or incapable of meeting the "
            "requirement.",
            "false",
            "103.N",
        )
    if k == 4:
        return Case(
            base + "A signed LOA states commercial service cannot meet the requirement; "
            "the contractor will reimburse travel, and the LOA lists the company billing "
            "address and a POC.",
            "true",
            "103.N",
        )
    return Case(
        base + "The technician says an LOA 'is on file somewhere'; its signature and "
        "contents have not been seen.",
        None,
        "103.N",
    )


def gen_booking_policy(c: Ctx) -> Case:
    w, r = c.who(), c.rng.choice([c.conus(), c.oconus()])
    k = c.k(6)
    if k == 0:
        return Case(
            f"{w[0].upper() + w[1:]} requested a seat on a military flight {r}, then had the "
            "TMC book a commercial flight as backup while the military request stays "
            "open.",
            "false",
            "102.B",
        )
    if k == 1:
        return Case(
            f"After the military airlift request {r} was formally cancelled, the TMC "
            f"booked {w} on a commercial flight.",
            "true",
            "102.B",
        )
    if k == 2:
        return Case(
            f"{w[0].upper() + w[1:]} booked two refundable flights {r} on different days to "
            "keep options open.",
            "false",
            "102.B",
        )
    if k == 3:
        return Case(
            f"A TMC is available, but {w} booked a flight {r} directly on the airline's website.",
            "false",
            "102.K",
        )
    if k == 4:
        return Case(
            f"{w[0].upper() + w[1:]} made all air and rental reservations {r} through the "
            "available TMC.",
            "true",
            "102.K",
        )
    return Case(
        f"{w[0].upper() + w[1:]} booked commercial travel {r}; an earlier request for military "
        "airlift may or may not still be open.",
        None,
        "102.B",
    )


def gen_rental_use(c: Ctx) -> Case:
    w = c.who()
    base = f"{w[0].upper() + w[1:]} on TDY has an authorized rental car and "
    k = c.k(8)
    good = [
        "drives between the hotel and the duty site",
        "drives to a nearby drugstore for prescriptions",
        "drives to a restaurant for dinner",
        "drives to a place of worship on Sunday",
        "drives to a dry cleaner",
    ]
    bad = [
        "drives 200 miles for a weekend beach trip",
        "drives to a theme park for a day of sightseeing",
        "lends the car to a friend for a personal errand",
    ]
    if k < 4:
        return Case(base + c.rng.choice(good) + ".", "true", "106.B")
    if k < 7:
        return Case(base + c.rng.choice(bad) + ".", "false", "106.B")
    return Case(
        base + "drives to a shopping mall; the purpose of the trip is not stated.", None, "106.B"
    )


def gen_ship_travel(c: Ctx) -> Case:
    w = c.who()
    base = f"{w[0].upper() + w[1:]} has PCS orders from Norfolk, VA to Rota, Spain. "
    k = c.k(7)
    if k == 0:
        return Case(
            base + "Command medical authority certified the traveler medically unable "
            "to fly; the order-issuing authority approved a U.S. registry ship.",
            "true",
            "107.A",
        )
    if k == 1:
        return Case(
            base + "The traveler would prefer a transatlantic cruise and asks to travel "
            "by ship; no extenuating circumstance exists.",
            "false",
            "107.A",
        )
    if k == 2:
        return Case(
            base + "Ship travel was approved for extenuating circumstances, but the "
            "traveler booked a foreign registry ship with no documentation that a U.S. "
            "registry ship was unavailable.",
            "false",
            "107.A",
        )
    if k == 3:
        return Case(
            base + "Ship travel was approved as more economical; no U.S. registry ship "
            "can serve the route, documented to the traveler, so a foreign registry ship "
            "is used.",
            "true",
            "107.A",
        )
    if k == 4:
        return Case(
            base + "The traveler booked a commercial ship without any authorization or "
            "approval from the order-issuing authority.",
            "false",
            "107.A",
        )
    if k == 5:
        return Case(
            base + "The order-issuing authority approved a U.S. registry ship because "
            "the travel can only be completed by ship.",
            "true",
            "107.A",
        )
    return Case(
        base + "The traveler has a private physician's note about fear of flying; "
        "whether command medical authority certified anything is unknown.",
        None,
        "107.A",
    )


# ---------------------------------------------------------------- score: compliance risk
RISK_LEVELS = [
    ["compliant", "Compliant: arrangement follows the DTR as described"],
    ["documentation_gap", "Documentation gap: allowed, but required documentation is missing"],
    [
        "excess_cost",
        "Excess cost: allowed only with the traveler paying costs above the "
        "Government-authorized amount",
    ],
    ["prohibited", "Prohibited: the DTR does not permit this arrangement"],
]
RISK_INSTR = [
    "Rate the DTR compliance risk of this travel arrangement.",
    "How serious is this arrangement's compliance risk under DTR Part I?",
    "Score the compliance risk, from compliant to prohibited.",
]
RISK_CASES: list[tuple[int, str, str]] = [
    (0, "{w} booked an economy seat {r} for {y} through the available TMC.", "102.K"),
    (
        0,
        "{w} has premium class approval from the premium class approval authority in the travel "
        "authorization for a documented disability, flying {r}.",
        "102.A",
    ),
    (
        0,
        "{w} rented the lowest-cost DTMO-agreement compact car through the TMC and drives only "
        "between lodging and the duty site.",
        "106.B",
    ),
    (
        0,
        "A group of 14 moving {r} uses a non-contract carrier chosen by the TO as best meeting "
        "the requirement.",
        "103.E",
    ),
    (
        1,
        "{w} has economy plus approval by email from the AO, but it was never added to the "
        "travel authorization for the flight {r}.",
        "102.A",
    ),
    (
        1,
        "A contractor's signed LOA for DoD airlift {r} states commercial service is unavailable "
        "but omits the emergency contact phone number.",
        "103.N",
    ),
    (
        1,
        "{w} flew a non-contract carrier {r} because no contract seats were available in time, "
        "but the exception has not yet been certified on any travel document.",
        "103.E",
    ),
    (
        2,
        "{w} was issued a business class ticket {r} without premium class approval documentation.",
        "102.A",
    ),
    (2, "{w} was approved for circuitous travel {r} with a stop in Lisbon on the way.", "102.J"),
    (
        2,
        "{w} skipped the City Pair flight {r} because the only confirmed pet space was on a "
        "costlier non-contract flight.",
        "103.E",
    ),
    (3, "{w} uses a standing blanket authorization for business class on all trips {r}.", "102.A"),
    (
        3,
        "{w} holds a commercial booking {r} while a military airlift request is still active.",
        "102.B",
    ),
    (3, "{w} bought a military furlough fare for official travel {r}.", "103.E"),
    (3, "A defense contractor bought a GSA City Pair fare for travel {r}.", "103.E"),
    (
        3,
        "{w} booked an intra-theater leg on an IASA Category 2 carrier with no GCC waiver.",
        "103.F",
    ),
]


def gen_compliance_risk(c: Ctx) -> Case:
    lvl, tmpl, sec = RISK_CASES[c.k(len(RISK_CASES))]
    text = tmpl.format(w=c.who(), r=c.rng.choice([c.conus(), c.oconus()]), y=c.why())
    return Case(text[0].upper() + text[1:], RISK_LEVELS[lvl][0], sec)


# ---------------------------------------------------------------- choice: governing provision
PROVISIONS = [
    ["102.A", "Class of service / premium class travel (Ch. 102, para A)"],
    ["102.B", "Dual commitment transportation (Ch. 102, para B)"],
    ["102.C", "Routing and mode selection (Ch. 102, para C)"],
    ["102.J", "Circuitous travel (Ch. 102, para J)"],
    ["102.K", "Travel Management Company use (Ch. 102, para K)"],
    ["103.E", "GSA City Pair Program fares and exceptions (Ch. 103, para E)"],
    ["103.F", "Foreign flag carriers (Ch. 103, para F)"],
    ["103.N", "Defense contractor personnel on DoD airlift (Ch. 103, para N)"],
    ["106.B", "Rental vehicles (Ch. 106, para B)"],
    ["107.A", "Commercial ship transportation (Ch. 107, para A)"],
]
ROUTE_INSTR = [
    "Which DTR Part I provision most directly governs this scenario?",
    "Cite the DTR Part I paragraph that controls this travel decision.",
]


@dataclass
class Family:
    task: str
    kind: str
    gen: Callable[[Ctx], Case]
    instructions: list[str]
    options: list[list[str]] | None = None


NOUL_INSTR = {
    "dtr_fare_eligibility": [
        "Is the fare described authorized for this travel under the DTR?",
        "May this traveler use the fare described?",
    ],
    "dtr_cpp_exception": [
        "Is use of the non-contract carrier on this City Pair route authorized?",
        "Does a valid GSA CPP exception apply here?",
    ],
    "dtr_foreign_carrier": [
        "Is use of this air carrier authorized for official DoD travel?",
        "Does DTR Part I permit this carrier for the leg described?",
    ],
    "dtr_contractor_airlift": [
        "Is this contractor eligible for transportation on DoD airlift?",
        "May the contractor be manifested on DoD airlift?",
    ],
    "dtr_booking_policy": [
        "Is this booking action permitted under DTR Part I?",
        "Does this booking comply with DTR booking rules?",
    ],
    "dtr_rental_use": [
        "Is this use of the rental car an authorized official purpose?",
        "Does the DTR permit this rental car use?",
    ],
    "dtr_ship_travel": [
        "Is commercial ship travel authorized at Government expense here?",
        "Does the DTR permit this ship travel?",
    ],
}
NOUL_GENS = {
    "dtr_fare_eligibility": gen_fare_eligibility,
    "dtr_cpp_exception": gen_cpp_exception,
    "dtr_foreign_carrier": gen_foreign_carrier,
    "dtr_contractor_airlift": gen_contractor_airlift,
    "dtr_booking_policy": gen_booking_policy,
    "dtr_rental_use": gen_rental_use,
    "dtr_ship_travel": gen_ship_travel,
}
FAMILIES: dict[str, Family] = {
    "dtr_class_of_service": Family(
        "dtr_class_of_service", "choice", gen_class_of_service, CLASS_INSTR, CLASS_OPTIONS
    ),
    "dtr_travel_mode": Family(
        "dtr_travel_mode", "choice", gen_travel_mode, MODE_INSTR, MODE_OPTIONS
    ),
    "dtr_compliance_risk": Family(
        "dtr_compliance_risk", "score", gen_compliance_risk, RISK_INSTR, RISK_LEVELS
    ),
    **{t: Family(t, "noul", g, NOUL_INSTR[t]) for t, g in NOUL_GENS.items()},
}
# Chapters 106 and 107 never appear in training: out-of-domain sections for --holdout.
HOLDOUT_TASKS = ("dtr_rental_use", "dtr_ship_travel")
ROUTING_TASK = "dtr_provision_routing"


def options_for(fam: Family) -> list[list[str]]:
    if fam.kind == "noul":
        return [["false", NOUL_DESC[0]], ["true", NOUL_DESC[1]]]
    return [list(o) for o in fam.options or []]


def case_examples(fam: Family, case: Case) -> list[Example]:
    opts = options_for(fam)
    names = [o[0] for o in opts]
    common = dict(
        kind=fam.kind,
        state=_state(case),
        instructions=fam.instructions[0],
        instruction_variants=fam.instructions[1:],
        options=opts,
        task=fam.task,
    )
    if case.label is None:
        # Both labels at half weight: a soft 0.5 target under cross-entropy.
        return [Example(label=i, weight=0.5, **common) for i in (0, 1)]
    return [Example(label=names.index(case.label), **common)]


def routing_example(case: Case) -> Example:
    return Example(
        kind="choice",
        state=_state(case, with_excerpt=False),
        instructions=ROUTE_INSTR[0],
        instruction_variants=ROUTE_INSTR[1:],
        options=[list(o) for o in PROVISIONS],
        label=[p[0] for p in PROVISIONS].index(case.section),
        task=ROUTING_TASK,
    )


def generate(task: str, n: int, *, seed: int = 0, half: int = 0) -> list[Example]:
    """`n` distinct scenarios for `task` (ambiguous ones yield two weighted rows)."""
    ctx = Ctx(half=half, rng=random.Random(f"{task}:{seed}:{half}"))
    seen: set[str] = set()
    out: list[Example] = []
    tries = 0
    while len(seen) < n and tries < n * 50:
        tries += 1
        if task == ROUTING_TASK:
            routable = [t for t in FAMILIES if t not in HOLDOUT_TASKS]
            pick = ctx.rng.randrange(len(routable)) if half == 0 else len(seen) % len(routable)
            fam = FAMILIES[routable[pick]]
            case = fam.gen(ctx)
        else:
            fam = FAMILIES[task]
            case = fam.gen(ctx)
        if case.scenario in seen:
            continue
        seen.add(case.scenario)
        out.extend([routing_example(case)] if task == ROUTING_TASK else case_examples(fam, case))
    return out


ALL_TASKS = [*FAMILIES, ROUTING_TASK]

# DTR-Bench: (task, scenarios) drawn from the bench vocabulary half.
BENCH_COUNTS = {
    "dtr_class_of_service": 15,
    "dtr_travel_mode": 12,
    "dtr_compliance_risk": 15,
    "dtr_fare_eligibility": 14,
    "dtr_cpp_exception": 14,
    "dtr_foreign_carrier": 14,
    "dtr_contractor_airlift": 8,
    "dtr_booking_policy": 12,
    "dtr_rental_use": 12,
    "dtr_ship_travel": 10,
    ROUTING_TASK: 16,
}
REVIEW_NOTE = (
    "Label derived from the cited DTR Part I excerpt by the scenario template and checked "
    "item-by-item against that excerpt by the prototype author (Devin, AI). Not yet reviewed "
    "by a DoD travel SME."
)


def bench_items(seed: int = 0) -> list[dict]:
    """DTR-Bench rows: one per scenario, `expected` is an option name or 'abstain'."""
    items: list[dict] = []
    for task, n in BENCH_COUNTS.items():
        rows = generate(task, n, seed=seed, half=1)
        by_scenario: dict[str, list[Example]] = {}
        for ex in rows:
            by_scenario.setdefault(ex.state["scenario"], []).append(ex)
        for i, exs in enumerate(by_scenario.values()):
            ex = exs[0]
            names = [o[0] for o in ex.options]
            # Score answers key probabilities by level index, so score gold is the index.
            gold = str(ex.label) if ex.kind == "score" else names[ex.label]
            expected = "abstain" if len(exs) > 1 else gold
            section = ex.state["dtr_citation"]
            q = ex.to_question().model_dump(exclude_none=True)
            items.append(
                {
                    "item_id": f"{task}-{i:03d}",
                    "task": task,
                    "kind": ex.kind,
                    "split": "holdout" if task in HOLDOUT_TASKS else "in_domain",
                    "state": ex.state,
                    "question": q,
                    "expected": expected,
                    "option_names": names,
                    "dtr_citation": section,
                    "review": {
                        "status": "self_reviewed",
                        "reviewer": "Devin (AI)",
                        "sme_review": "pending",
                        "note": REVIEW_NOTE,
                    },
                }
            )
    return items


def question_for(fam: Family) -> dict:
    """The serving-API question for a family, exactly as DTR-Bench poses it."""
    ex = Example(
        kind=fam.kind,
        state={},
        instructions=fam.instructions[0],
        options=options_for(fam),
        label=0,
        task=fam.task,
    )
    return ex.to_question().model_dump(exclude_none=True)


def catalog() -> dict:
    """What an app needs to ask the decider about a new scenario: per determination
    type, the question, its governing DTR excerpt and citation, and the option labels."""
    sections = {
        "dtr_class_of_service": "102.A",
        "dtr_travel_mode": "102.C",
        "dtr_fare_eligibility": "103.E",
        "dtr_cpp_exception": "103.E",
        "dtr_foreign_carrier": "103.F",
        "dtr_contractor_airlift": "103.N",
        "dtr_booking_policy": "102.K",
        "dtr_rental_use": "106.B",
        "dtr_ship_travel": "107.A",
        "dtr_compliance_risk": "102.A",
    }
    out = {}
    for task, fam in FAMILIES.items():
        cite, text = EXCERPTS[sections[task]]
        out[task] = {
            "kind": fam.kind,
            "section": sections[task],
            "dtr_citation": cite,
            "dtr_excerpt": text,
            "question": question_for(fam),
            "option_names": [o for o, _ in options_for(fam)],
            "holdout": task in HOLDOUT_TASKS,
        }
    return {"source": SOURCE, "tasks": out}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Write the DTR-Bench gold set as JSONL.")
    ap.add_argument("--out", default="data/dtr_bench.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", help="Also write the app's question catalog (JSON) here.")
    args = ap.parse_args(argv)
    if args.catalog:
        with open(args.catalog, "w", encoding="utf-8") as fh:
            json.dump(catalog(), fh, ensure_ascii=False, indent=1)
        print(f"wrote catalog -> {args.catalog}")
    items = bench_items(args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} DTR-Bench items -> {args.out}")


if __name__ == "__main__":
    main()
