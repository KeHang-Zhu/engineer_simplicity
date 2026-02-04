"""
Deferred Acceptance (DA) Simulation with LLMs

This module implements Deferred Acceptance matching mechanisms using LLM agents.
Supports both direct revelation (full ranking submission) and OSP (sequential local queries).

Architecture mirrors util_plan.py (auction simulation).
"""

import random
import json
import os
import re
from edsl import Model, Survey, Cache
from edsl.questions import QuestionRank, QuestionMultipleChoice, QuestionFreeText
from edsl.prompts import Prompt

current_script_path = os.path.dirname(os.path.abspath(__file__))
prompt_dir = os.path.join(current_script_path, '../Prompt/')


def save_json(data, filename, directory):
    """Save data to a JSON file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    return file_path


class Rule_DA:
    """
    Defines DA mechanism rules and renders Jinja2 templates.
    """
    def __init__(self, mechanism_type, intervention_type="baseline",
                 special_name=None, templates_dir=None, common_range=[40, 70],
                 private_range=20, global_ranking_strategy="average"):
        """
        Initialize DA rule.

        Args:
            mechanism_type: "direct" or "osp"
            intervention_type: Cognitive intervention (e.g., "baseline", "axis1_enumerate")
            special_name: Override template filename
            templates_dir: Path to rule_template/DA/
            common_range: Range for common value component
            private_range: Range for private value component
            global_ranking_strategy: Strategy for computing global ranking
                - "average": Based on average values
                - "fixed": Fixed ranking for all experiments
                - "random": Random ranking
                - "misleading": Reverse of average (for experiments)
        """
        self.mechanism_type = mechanism_type
        self.intervention_type = intervention_type
        self.special_name = special_name
        self.common_range = common_range
        self.private_range = private_range
        self.global_ranking_strategy = global_ranking_strategy
        self.number_students = 4
        self.number_schools = 4

        # Resolve templates directory
        if templates_dir is None:
            self.templates_dir = os.path.join(current_script_path, '../rule_template/DA/')
        elif not os.path.isabs(templates_dir):
            self.templates_dir = os.path.join(current_script_path, '..', templates_dir)
        else:
            self.templates_dir = templates_dir

        # Load template
        self.rule_explanation = self._load_template()

    def _load_template(self):
        """Load appropriate template file based on mechanism type."""
        if self.special_name:
            template_file = self.special_name
        else:
            # Default templates
            if self.mechanism_type == "direct":
                if self.intervention_type == "baseline":
                    template_file = "da_direct_traditional.txt"
                else:
                    template_file = f"{self.intervention_type}.txt"
            elif self.mechanism_type == "osp":
                template_file = "da_osp_choice.txt"
            else:
                raise ValueError(f"Unknown mechanism_type: {self.mechanism_type}")

        template_path = os.path.join(self.templates_dir, template_file)
        template_string = Prompt.from_txt(template_path)

        # Return template string (will be rendered later with student-specific data)
        return template_string

    def describe(self):
        """Print mechanism description."""
        print(f"DA Mechanism Type: {self.mechanism_type}")
        print(f"Intervention Type: {self.intervention_type}")
        print(f"Number of Students: {self.number_students}")
        print(f"Number of Schools: {self.number_schools}")


class Student:
    """
    Represents a student agent in the DA matching mechanism.
    Parallel to Bidder class in auction simulation.
    """
    def __init__(self, value_dict, priority_dict, name, rule):
        """
        Initialize student.

        Args:
            value_dict: Dict[school, value] - values for each school
            priority_dict: Dict[school, priority_rank] - priorities at each school
            name: Student name ("A", "B", "C", "D")
            rule: Rule_DA instance
        """
        self.name = f"Student {name}"
        self.rule = rule
        self.values = value_dict  # {"w": 85, "x": 72, "y": 90, "z": 65}
        self.priorities = priority_dict  # {"w": 1, "x": 2, "y": 1, "z": 2}

        # Outcomes
        self.submitted_ranking = None  # For direct: ["w", "x", "y", "z"]
        self.osp_choices = []  # For OSP: sequential choices
        self.matched_school = None  # Final match
        self.utility = 0  # Final utility
        self.reasoning = ""  # LLM reasoning

    def get_utility(self, matched_school):
        """Return value for matched school or 0 if unmatched."""
        if matched_school is None:
            return 0
        return self.values.get(matched_school, 0)

    def __repr__(self):
        return f"{self.name}(values={self.values}, matched={self.matched_school})"


class DA_Direct:
    """
    Implements direct revelation mechanism (submit full ranking once).
    Uses QuestionRank for parallel querying of all students.
    """
    def __init__(self, students, rule, model, cache=None, global_ranking=None):
        """
        Initialize direct revelation mechanism.

        Args:
            students: List of Student objects
            rule: Rule_DA instance
            model: EDSL Model instance
            cache: EDSL Cache instance
            global_ranking: Optional global ranking string for social information
        """
        self.students = students
        self.rule = rule
        self.model = model
        self.cache = cache
        self.global_ranking = global_ranking or "w > x > y > z"  # Default fallback
        self.da_trace = []  # Trace of DA algorithm execution

    def run(self):
        """
        Main execution: parallel Survey with QuestionFreeText to collect reasoning.

        Returns:
            Dict with 'rankings', 'reasoning', 'matches', 'da_trace'
        """
        print("Running DA Direct Mechanism...")

        # STEP 1: Build prompts for all students
        student_prompts = []
        for student in self.students:
            prompt = self._build_student_prompt(student)
            student_prompts.append((student, prompt))

        # STEP 2: Create parallel Survey with QuestionFreeText (to collect reasoning)
        questions = []
        for student, prompt in student_prompts:
            q_freetext = QuestionFreeText(
                question_name=f"q_reason_{student.name.replace(' ', '_')}",
                question_text=prompt
            )
            questions.append(q_freetext)

        # STEP 3: Execute parallel LLM calls
        survey = Survey(questions=questions)
        result = survey.by(self.model).run(cache=self.cache)

        # STEP 4: Parse reasoning and rankings with retry logic
        submitted_rankings = {}
        reasoning_dict = {}

        for i, (student, prompt) in enumerate(student_prompts):
            question_name = f"q_reason_{student.name.replace(' ', '_')}"
            response = result.select(question_name).to_list()[0]

            # Parse reasoning and ranking with retries
            reason, ranking = self._parse_and_validate_response(response, student, prompt)

            submitted_rankings[student.name] = ranking
            reasoning_dict[student.name] = reason

            student.submitted_ranking = ranking
            student.reasoning = reason

            print(f"{student.name} submitted ranking: {ranking}")

        # STEP 5: Compute truthfulness
        truthfulness = self._compute_truthfulness()

        # STEP 6: Run DA algorithm
        matches = self._run_da_algorithm(submitted_rankings)

        # STEP 7: Record outcomes
        self._record_outcomes(matches)

        return {
            'rankings': submitted_rankings,
            'reasoning': reasoning_dict,
            'truthfulness': truthfulness,
            'matches': matches,
            'da_trace': self.da_trace
        }

    def _build_student_prompt(self, student):
        """Render template with student's values and priorities."""
        # Render main mechanism explanation template
        main_prompt = self.rule.rule_explanation.render({
            "student_id": student.name.split()[-1],  # "A", "B", etc.
            "vw": student.values["w"],
            "vx": student.values["x"],
            "vy": student.values["y"],
            "vz": student.values["z"],
            "pw": student.priorities["w"],
            "px": student.priorities["x"],
            "py": student.priorities["y"],
            "pz": student.priorities["z"],
            "global_ranking": self.global_ranking  # Add global ranking
        })

        # Load and append da_ask.txt (reasoning instruction)
        da_ask_path = os.path.join(prompt_dir, 'da_ask.txt')
        if os.path.exists(da_ask_path):
            with open(da_ask_path, 'r') as f:
                da_ask_content = f.read()
            full_prompt = str(main_prompt) + "\n\n" + da_ask_content
        else:
            # Fallback if da_ask.txt doesn't exist
            full_prompt = str(main_prompt)

        return full_prompt

    def _parse_and_validate_response(self, initial_response, student, full_prompt):
        """
        Parse <REASON> and <DECISION> tags with 3-attempt retry logic.

        Args:
            initial_response: LLM response with <REASON> and <DECISION> tags
            student: Student object
            full_prompt: Full prompt text for retry

        Returns:
            Tuple[str, List[str]]: (reason, ranking)

        Raises:
            RuntimeError: If parsing fails after 3 attempts
        """
        response = initial_response

        for attempt in range(3):
            try:
                # Parse REASON and DECISION tags
                reason, decision_text = self._parse_reason_decision(response)

                # Use gpt-4o-mini to extract ranking from decision text
                ranking = self._extract_ranking_with_model(decision_text, student)

                if self._is_valid_ranking(ranking):
                    return reason, ranking
                raise ValueError("Invalid ranking: missing schools or duplicates")

            except Exception as e:
                print(f"{student.name} parsing error (attempt {attempt+1}/3): {e}")

                if attempt < 2:
                    # Retry with error message
                    q_retry = QuestionFreeText(
                        question_name="q_reason_retry",
                        question_text=full_prompt + f"\n\nError: {e}. You MUST use the format:\n<REASON>Your reasoning here</REASON>\n<DECISION>Ranking: w > x > y > z</DECISION>"
                    )
                    survey = Survey(questions=[q_retry])
                    result = survey.by(self.model).run(cache=self.cache)
                    response = result.select("q_reason_retry").to_list()[0]

        raise RuntimeError(f"{student.name} failed to parse response after 3 attempts")

    def _parse_reason_decision(self, text):
        """
        Parse <REASON> and <DECISION> tags from LLM response.

        Args:
            text: LLM response text

        Returns:
            Tuple[str, str]: (reason, decision_text)

        Raises:
            ValueError: If tags not found
        """
        # Parse REASON
        reason_pattern = r"<REASON>(.*?)</REASON>"
        reason_match = re.search(reason_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not reason_match:
            raise ValueError("REASON tag not found")
        reason = reason_match.group(1).strip()

        # Parse DECISION
        decision_pattern = r"<DECISION>(.*?)</DECISION>"
        decision_match = re.search(decision_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not decision_match:
            raise ValueError("DECISION tag not found")
        decision = decision_match.group(1).strip()

        return reason, decision

    def _extract_ranking_with_model(self, decision_text, student):
        """
        Use gpt-4o-mini to extract ranking from decision text.

        Args:
            decision_text: Text from <DECISION> tag
            student: Student object

        Returns:
            List[str]: Ranking ["w", "x", "y", "z"]
        """
        # First try direct parsing
        try:
            ranking = self._parse_ranking(decision_text)
            if self._is_valid_ranking(ranking):
                return ranking
        except:
            pass

        # If direct parsing fails, use gpt-4o-mini
        extraction_prompt = f"""
Extract the school ranking from the following decision text.
The student must rank 4 schools: w, x, y, z.

Decision text:
{decision_text}

Respond with ONLY the ranking in the format: w > x > y > z
Do NOT include any other text.
"""

        q_extract = QuestionFreeText(
            question_name="q_extract",
            question_text=extraction_prompt
        )
        survey = Survey(questions=[q_extract])

        # Use gpt-4o-mini for extraction
        extract_model = Model("gpt-4o-mini", temperature=0)
        result = survey.by(extract_model).run()
        extracted = result.select("q_extract").to_list()[0]

        # Parse the extracted text
        ranking = self._parse_ranking(extracted)
        return ranking

    def _parse_and_validate_ranking(self, initial_response, student, full_prompt):
        """
        Parse ranking with 3-attempt retry logic.

        Args:
            initial_response: LLM response
            student: Student object
            full_prompt: Full prompt text for retry

        Returns:
            List[str]: Validated ranking ["w", "x", "y", "z"]

        Raises:
            RuntimeError: If parsing fails after 3 attempts
        """
        response = initial_response

        for attempt in range(3):
            try:
                ranking = self._parse_ranking(response)
                if self._is_valid_ranking(ranking):
                    return ranking
                raise ValueError("Invalid ranking: missing schools or duplicates")
            except Exception as e:
                print(f"{student.name} parsing error (attempt {attempt+1}/3): {e}")

                if attempt < 2:
                    # Retry with error message
                    q_retry = QuestionRank(
                        question_name="q_rank_retry",
                        question_text=full_prompt + f"\n\nError: {e}. You MUST follow the format: Ranking: <1st> > <2nd> > <3rd> > <4th>",
                        question_options=["w", "x", "y", "z"]
                    )
                    survey = Survey(questions=[q_retry])
                    result = survey.by(self.model).run(cache=self.cache)
                    response = result.select("q_rank_retry").to_list()[0]

        raise RuntimeError(f"{student.name} failed to parse ranking after 3 attempts")

    def _parse_ranking(self, response):
        """
        Parse ranking from QuestionRank response.

        Handles formats:
        - ["w", "x", "y", "z"] (list)
        - "Ranking: w > x > y > z" (text)
        - "w > x > y > z" (text)

        Returns:
            List[str]: ["w", "x", "y", "z"]
        """
        if isinstance(response, list):
            return [str(x).lower().strip() for x in response]

        # Parse text format
        text = str(response).lower()

        # Try "Ranking: w > x > y > z"
        pattern = r"ranking:\s*([w-z])\s*>\s*([w-z])\s*>\s*([w-z])\s*>\s*([w-z])"
        match = re.search(pattern, text)
        if match:
            return list(match.groups())

        # Try direct "w > x > y > z"
        pattern2 = r"([w-z])\s*>\s*([w-z])\s*>\s*([w-z])\s*>\s*([w-z])"
        match2 = re.search(pattern2, text)
        if match2:
            return list(match2.groups())

        # Try comma-separated "w, x, y, z"
        pattern3 = r"([w-z]),\s*([w-z]),\s*([w-z]),\s*([w-z])"
        match3 = re.search(pattern3, text)
        if match3:
            return list(match3.groups())

        raise ValueError(f"Could not parse ranking from: {response}")

    def _is_valid_ranking(self, ranking):
        """Validate ranking contains all schools exactly once."""
        if not isinstance(ranking, list) or len(ranking) != 4:
            return False
        schools = {"w", "x", "y", "z"}
        return set(ranking) == schools

    def _run_da_algorithm(self, submitted_rankings):
        """
        Run student-proposing Deferred Acceptance algorithm.

        Args:
            submitted_rankings: Dict[student_name, List[school]]

        Returns:
            Dict[student_name, Optional[school]]: Final matches
        """
        print("\nRunning DA Algorithm...")

        tentative_matches = {}  # {school: student_name}
        student_next_proposal = {s.name: 0 for s in self.students}
        unmatched = set(s.name for s in self.students)

        round_num = 0
        while unmatched:
            proposals = {}  # {school: [students]}

            # Each unmatched student proposes to next school on their list
            for student_name in list(unmatched):
                idx = student_next_proposal[student_name]
                ranking = submitted_rankings[student_name]

                if idx >= len(ranking):
                    # Student exhausted their list
                    unmatched.remove(student_name)
                    continue

                school = ranking[idx]
                student_next_proposal[student_name] += 1
                proposals.setdefault(school, []).append(student_name)

            if not proposals:
                break  # No more proposals possible

            # Each school keeps highest-priority proposer
            rejections = []
            for school, proposers in proposals.items():
                candidates = proposers[:]
                if school in tentative_matches:
                    candidates.append(tentative_matches[school])

                best = self._select_by_priority(school, candidates)

                # Track rejections
                for candidate in candidates:
                    if candidate != best:
                        rejections.append((candidate, school))
                        if candidate in proposers:
                            # Student was just rejected, remains unmatched
                            pass

                # Update tentative match
                if school in tentative_matches and tentative_matches[school] != best:
                    old_match = tentative_matches[school]
                    unmatched.add(old_match)

                tentative_matches[school] = best
                unmatched.discard(best)

            # Log this round
            self.da_trace.append({
                'round': round_num,
                'proposals': proposals.copy(),
                'tentative_matches': tentative_matches.copy(),
                'rejections': rejections
            })

            print(f"  Round {round_num}: {len(proposals)} proposals, {len(rejections)} rejections")
            round_num += 1

        # Convert school->student to student->school
        matches = {s.name: None for s in self.students}
        for school, student_name in tentative_matches.items():
            matches[student_name] = school

        print(f"\nDA Algorithm complete after {round_num} rounds")
        return matches

    def _select_by_priority(self, school, candidates):
        """
        Select student with highest priority at school.
        Lower priority number = higher priority (1 is best).

        Args:
            school: School name
            candidates: List of student names

        Returns:
            str: Best student name
        """
        best = None
        best_priority = float('inf')

        for student_name in candidates:
            student = next(s for s in self.students if s.name == student_name)
            priority = student.priorities[school]

            if priority < best_priority:
                best_priority = priority
                best = student_name

        return best

    def _compute_truthfulness(self):
        """
        Compute truthfulness for each student by comparing true preferences with submitted ranking.

        Returns:
            Dict[student_name, bool]: Truthfulness for each student
        """
        truthfulness = {}

        for student in self.students:
            # Compute true preference ranking from values (descending order)
            true_ranking = sorted(
                student.values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            true_ranking = [school for school, _ in true_ranking]

            # Compare with submitted ranking
            submitted = student.submitted_ranking

            # Truthful if rankings match exactly
            is_truthful = (true_ranking == submitted)

            truthfulness[student.name] = is_truthful

            if not is_truthful:
                print(f"  {student.name} MISREPORTED:")
                print(f"    True preference: {' > '.join(true_ranking)}")
                print(f"    Submitted: {' > '.join(submitted)}")

        return truthfulness

    def _record_outcomes(self, matches):
        """Store outcomes in student objects."""
        for student in self.students:
            student.matched_school = matches[student.name]
            student.utility = student.get_utility(student.matched_school)
            print(f"{student.name}: matched to {student.matched_school}, utility={student.utility}")


class DA_OSP:
    """
    Implements OSP mechanism (sequential local queries).
    Uses QuestionMultipleChoice with dynamic available sets.
    """
    def __init__(self, students, rule, model, cache=None, global_ranking=None):
        """
        Initialize OSP mechanism.

        Args:
            students: List of Student objects
            rule: Rule_DA instance
            model: EDSL Model instance
            cache: EDSL Cache instance
            global_ranking: Optional global ranking string for social information
        """
        self.students = students
        self.rule = rule
        self.model = model
        self.cache = cache
        self.global_ranking = global_ranking or "w > x > y > z"  # Default fallback

        # State management
        self.available_sets = {}  # {student_name: set of schools}
        self.tentative_matches = {}  # {school: student_name}
        self.osp_round = 0
        self.osp_history = []

    def run(self):
        """
        Main OSP loop: sequential rounds until all matched.

        Returns:
            Dict with 'osp_history', 'matches'
        """
        print("Running DA OSP Mechanism...")

        # Initialize: all schools available for all students
        for student in self.students:
            self.available_sets[student.name] = {"w", "x", "y", "z"}

        # Run OSP rounds until all matched
        while not self._all_students_matched():
            self._run_one_osp_round()
            self.osp_round += 1

        # Finalize matches
        matches = self._get_final_matches()
        self._record_outcomes(matches)

        # Compute truthfulness for OSP
        truthfulness = self._compute_osp_truthfulness()

        return {
            'osp_history': self.osp_history,
            'matches': matches,
            'truthfulness': truthfulness
        }

    def _all_students_matched(self):
        """Check if all students have been matched."""
        for student_name in self.available_sets:
            if len(self.available_sets[student_name]) > 0:
                # Check if student is actually matched
                if student_name not in self.tentative_matches.values():
                    return False
        return True

    def _run_one_osp_round(self):
        """Execute one round of parallel OSP queries."""
        print(f"\nOSP Round {self.osp_round}...")

        # STEP 1: Build prompts for students with available schools
        student_prompts = []
        for student in self.students:
            if len(self.available_sets[student.name]) > 0:
                # Only query if student has available schools and not yet matched
                if student.name not in self.tentative_matches.values():
                    prompt = self._build_osp_prompt(student)
                    student_prompts.append((student, prompt))

        if not student_prompts:
            return  # No students to query

        # STEP 2: Create parallel Survey with QuestionMultipleChoice
        questions = []
        for student, prompt in student_prompts:
            available = sorted(list(self.available_sets[student.name]))

            q_choice = QuestionMultipleChoice(
                question_name=f"q_osp_{student.name.replace(' ', '_')}_r{self.osp_round}",
                question_text=prompt,
                question_options=available
            )
            questions.append(q_choice)

        # STEP 3: Execute parallel LLM calls
        survey = Survey(questions=questions)
        result = survey.by(self.model).run(cache=self.cache)

        # STEP 4: Parse choices
        choices = {}
        for student, prompt in student_prompts:
            question_name = f"q_osp_{student.name.replace(' ', '_')}_r{self.osp_round}"
            response = result.select(question_name).to_list()[0]

            # Validate choice
            choice = self._validate_choice(response, student)
            choices[student.name] = choice
            student.osp_choices.append(choice)

            print(f"  {student.name} chose: {choice}")

        # STEP 5: Process choices via DA step
        self._process_osp_choices(choices)

    def _build_osp_prompt(self, student):
        """Render OSP template with dynamic available set."""
        available_list = sorted(self.available_sets[student.name])
        available_str = ", ".join(available_list)

        prompt = self.rule.rule_explanation.render({
            "student_id": student.name.split()[-1],
            "available_set": available_str,
            "vw": student.values["w"],
            "vx": student.values["x"],
            "vy": student.values["y"],
            "vz": student.values["z"],
            "global_ranking": self.global_ranking  # Add global ranking
        })
        return str(prompt)

    def _validate_choice(self, response, student):
        """
        Validate OSP choice is in available set.

        Args:
            response: LLM response
            student: Student object

        Returns:
            str: Validated school choice

        Raises:
            ValueError: If choice not in available set
        """
        available = self.available_sets[student.name]

        # Parse response
        if isinstance(response, str):
            choice = response.lower().strip()

            # Handle "Choice: w" format
            match = re.search(r"choice:\s*([w-z])", choice)
            if match:
                choice = match.group(1)

            if choice in available:
                return choice

        raise ValueError(f"Invalid choice '{response}' not in available set {available}")

    def _process_osp_choices(self, choices):
        """
        Process OSP choices: run DA step and update state.

        Args:
            choices: Dict[student_name, school]
        """
        # Save available sets BEFORE processing (for truthfulness checking)
        available_before = {k: sorted(list(v)) for k, v in self.available_sets.items()}

        proposals = {}  # {school: [students]}

        # Group proposals by school
        for student_name, school in choices.items():
            proposals.setdefault(school, []).append(student_name)

        rejections = []

        # Each school processes proposals
        for school, proposers in proposals.items():
            # Include current match if exists
            candidates = proposers[:]
            if school in self.tentative_matches:
                candidates.append(self.tentative_matches[school])

            # Select best by priority
            best = self._select_by_priority(school, candidates)

            # Track rejections
            for candidate in candidates:
                if candidate != best:
                    rejections.append((candidate, school))

            # Update tentative match
            old_match = self.tentative_matches.get(school)
            if old_match and old_match != best:
                rejections.append((old_match, school))

            self.tentative_matches[school] = best

        # Update available sets: remove rejected schools
        for student_name, school in rejections:
            self.available_sets[student_name].discard(school)

        # Remove matched students' available sets
        matched_students = set(self.tentative_matches.values())
        for student_name in matched_students:
            if student_name in self.available_sets:
                self.available_sets[student_name] = set()

        # Log this round (use available_before for truthfulness checking)
        self.osp_history.append({
            'round': self.osp_round,
            'choices': choices.copy(),
            'rejections': rejections,
            'tentative_matches': self.tentative_matches.copy(),
            'available_sets_before': available_before,  # Available when choices were made
            'available_sets_after': {k: sorted(list(v)) for k, v in self.available_sets.items()}  # After processing
        })

        print(f"  Tentative matches: {self.tentative_matches}")
        print(f"  Rejections: {len(rejections)}")

    def _select_by_priority(self, school, candidates):
        """Select student with highest priority at school."""
        best = None
        best_priority = float('inf')

        for student_name in candidates:
            student = next(s for s in self.students if s.name == student_name)
            priority = student.priorities[school]

            if priority < best_priority:
                best_priority = priority
                best = student_name

        return best

    def _get_final_matches(self):
        """Convert school->student to student->school."""
        matches = {s.name: None for s in self.students}
        for school, student_name in self.tentative_matches.items():
            matches[student_name] = school
        return matches

    def _record_outcomes(self, matches):
        """Store outcomes in student objects."""
        for student in self.students:
            student.matched_school = matches[student.name]
            student.utility = student.get_utility(student.matched_school)
            print(f"{student.name}: matched to {student.matched_school}, utility={student.utility}")

    def _compute_osp_truthfulness(self):
        """
        Compute truthfulness for OSP mechanism.

        For OSP, a student is truthful if in EVERY round they chose their
        most preferred school among the available options.

        Returns:
            Dict[student_name, bool]: Truthfulness for each student
        """
        truthfulness = {}

        for student in self.students:
            is_truthful = True

            # Check each round in OSP history
            for round_data in self.osp_history:
                round_num = round_data['round']
                # Use available_sets_before (what was available when choice was made)
                available_set = set(round_data.get('available_sets_before',
                                                   round_data.get('available_sets', [])).get(student.name, []))

                # Skip if no available schools (student already matched)
                if not available_set:
                    continue

                # Get student's choice in this round
                if student.name in round_data['choices']:
                    choice = round_data['choices'][student.name]

                    # Find most preferred school in available set
                    available_values = {school: student.values[school]
                                      for school in available_set}
                    best_school = max(available_values.items(), key=lambda x: x[1])[0]

                    # Check if choice matches best school
                    if choice != best_school:
                        is_truthful = False
                        print(f"{student.name} MISREPORTED in round {round_num}:")
                        print(f"  Available: {sorted(available_set)}")
                        print(f"  Values: {available_values}")
                        print(f"  Best choice: {best_school} (${student.values[best_school]})")
                        print(f"  Submitted: {choice} (${student.values[choice]})")
                        break

            truthfulness[student.name] = is_truthful

        # Calculate overall truthfulness rate
        truthful_count = sum(truthfulness.values())
        total_count = len(truthfulness)
        truthfulness_rate = truthful_count / total_count if total_count > 0 else 0

        print(f"\nOSP Truthfulness rate: {truthfulness_rate:.1%}")

        return truthfulness


class DA_plan:
    """
    Orchestrates DA experiments.
    Parallel to Auction_plan class.
    """
    def __init__(self, number_students, number_schools, rule, output_dir,
                 timestring=None, cache=None, model='gpt-4o', temperature=0,
                 service_name=None, config_dict=None, experiment_index=None):
        """
        Initialize DA plan.

        Args:
            number_students: Number of students (fixed at 4)
            number_schools: Number of schools (fixed at 4)
            rule: Rule_DA instance
            output_dir: Output directory for results (base folder, not run_*)
            timestring: Timestamp string for filenames
            cache: EDSL Cache instance
            model: Model name
            temperature: LLM temperature
            service_name: Optional service name for Model
            config_dict: Optional config dictionary to save
            experiment_index: Index of this experiment (for numbering)
        """
        import pandas as pd

        self.rule = rule
        self.students = []
        self.number_students = number_students
        self.number_schools = number_schools

        # Initialize model
        if service_name:
            self.model = Model(model, temperature=temperature, service_name=service_name)
        else:
            self.model = Model(model, temperature=temperature)

        self.cache = cache
        self.output_dir = output_dir  # No run_{timestamp} subfolder
        self.experiment_index = experiment_index

        # Generate timestring if not provided
        if timestring is None:
            timestring = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.timestring = timestring

        # Create subdirectories (only once)
        self.raw_data_dir = os.path.join(self.output_dir, "raw_data")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.prompts_dir = os.path.join(self.output_dir, "prompts")

        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.prompts_dir, exist_ok=True)

        # Save config if provided (only once, check if already exists)
        self.config_dict = config_dict
        config_path = os.path.join(self.output_dir, "config.yaml")
        if config_dict and not os.path.exists(config_path):
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            print(f"Config saved to: {config_path}")

        # Data storage
        self.values_list = {}  # {school: [values per student]}
        self.priorities_structure = None  # Fixed acyclic priorities
        self.data_to_save = {}

    def draw_values(self, seed=1234):
        """
        Generate values using common + private structure.
        Similar to affiliated value auctions.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        print(f"\nGenerating values with seed={seed}...")

        self.values_list = {school: [] for school in ["w", "x", "y", "z"]}

        for school in ["w", "x", "y", "z"]:
            # Draw common value for this school
            common_value = random.randint(self.rule.common_range[0],
                                          self.rule.common_range[1])

            # Each student gets common + private shock
            for student_idx in range(self.number_students):
                private_shock = random.randint(0, self.rule.private_range)
                total_value = common_value + private_shock
                self.values_list[school].append(total_value)

            print(f"  School {school}: common={common_value}, values={self.values_list[school]}")

        # Generate fixed acyclic priorities
        self.priorities_structure = self._generate_acyclic_priorities()
        print(f"\nFixed acyclic priorities:")
        for school, prios in self.priorities_structure.items():
            print(f"  {school}: {prios}")

        # Generate global ranking (social information)
        self.global_ranking = self._compute_global_ranking(
            strategy=self.rule.global_ranking_strategy
        )
        print(f"\nGlobal ranking ({self.rule.global_ranking_strategy}): {self.global_ranking}")

    def _generate_acyclic_priorities(self):
        """
        Return fixed Ergin-acyclic priority structure.
        From readme: Top-2 = {A, B}, Bottom-2 = {C, D}

        Returns:
            Dict[school, Dict[student_name, priority_rank]]
        """
        return {
            "w": {"Student A": 1, "Student B": 2, "Student C": 3, "Student D": 4},
            "x": {"Student B": 1, "Student A": 2, "Student C": 3, "Student D": 4},
            "y": {"Student A": 1, "Student B": 2, "Student D": 3, "Student C": 4},
            "z": {"Student B": 1, "Student A": 2, "Student D": 3, "Student C": 4}
        }

    def _compute_global_ranking(self, strategy="average"):
        """
        Compute global ranking of schools to provide social information.

        Args:
            strategy: How to compute ranking
                - "average": Based on average values across students
                - "truthful": Based on actual student preferences (if known)
                - "fixed": Fixed ranking for all experiments
                - "random": Random ranking
                - "misleading": Reverse of average (for experiments)

        Returns:
            str: Ranking string like "y > x > w > z"
        """
        if strategy == "average":
            # Compute average value for each school
            avg_values = {}
            for school in ["w", "x", "y", "z"]:
                avg_values[school] = sum(self.values_list[school]) / len(self.values_list[school])

            # Sort by average value (descending)
            sorted_schools = sorted(avg_values.items(), key=lambda x: x[1], reverse=True)
            ranking = " > ".join([school for school, _ in sorted_schools])

            print(f"  Average values: {avg_values}")
            return ranking

        elif strategy == "fixed":
            # Fixed ranking (can be used as control condition)
            return "y > x > w > z"

        elif strategy == "random":
            # Random ranking
            schools = ["w", "x", "y", "z"]
            random.shuffle(schools)
            return " > ".join(schools)

        elif strategy == "misleading":
            # Reverse of average (for testing effects of misinformation)
            avg_values = {}
            for school in ["w", "x", "y", "z"]:
                avg_values[school] = sum(self.values_list[school]) / len(self.values_list[school])

            sorted_schools = sorted(avg_values.items(), key=lambda x: x[1], reverse=False)  # Ascending!
            ranking = " > ".join([school for school, _ in sorted_schools])

            print(f"  ⚠️  Misleading ranking (reversed)")
            return ranking

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def build_students(self):
        """Create Student instances with values and priorities."""
        student_names = ["A", "B", "C", "D"]

        print(f"\nBuilding {self.number_students} students...")

        for i in range(self.number_students):
            # Extract values for this student
            value_dict = {}
            for school in ["w", "x", "y", "z"]:
                value_dict[school] = self.values_list[school][i]

            # Extract priorities for this student
            priority_dict = {}
            student_name = f"Student {student_names[i]}"
            for school in ["w", "x", "y", "z"]:
                priority_dict[school] = self.priorities_structure[school][student_name]

            student = Student(
                value_dict=value_dict,
                priority_dict=priority_dict,
                name=student_names[i],
                rule=self.rule
            )
            self.students.append(student)

            print(f"  {student.name}: values={value_dict}, priorities={priority_dict}")

    def run(self):
        """
        Execute one DA round.
        Dispatch to DA_Direct or DA_OSP based on mechanism type.

        Returns:
            Dict with results
        """
        print(f"\n{'='*70}")
        print(f"Running DA Mechanism: {self.rule.mechanism_type}")
        print(f"{'='*70}")

        if self.rule.mechanism_type == "direct":
            da_mechanism = DA_Direct(
                students=self.students,
                rule=self.rule,
                model=self.model,
                cache=self.cache,
                global_ranking=self.global_ranking  # Pass global ranking
            )
            results = da_mechanism.run()

        elif self.rule.mechanism_type == "osp":
            da_mechanism = DA_OSP(
                students=self.students,
                rule=self.rule,
                model=self.model,
                cache=self.cache,
                global_ranking=self.global_ranking  # Pass global ranking
            )
            results = da_mechanism.run()

        else:
            raise ValueError(f"Unknown mechanism: {self.rule.mechanism_type}")

        # Store results
        self._record_results(results)

        return results

    def _record_results(self, results):
        """Store results in data structure."""
        matches = results['matches']

        # Build data structure
        self.data_to_save = {
            "mechanism_type": self.rule.mechanism_type,
            "global_ranking": self.global_ranking,  # Add global ranking info
            "global_ranking_strategy": self.rule.global_ranking_strategy,  # Add strategy info
            "values": {s.name: s.values for s in self.students},
            "priorities": {s.name: s.priorities for s in self.students},
            "matches": matches,
            "utilities": {s.name: s.utility for s in self.students}
        }

        # Add mechanism-specific data
        if self.rule.mechanism_type == "direct":
            self.data_to_save["rankings"] = {s.name: s.submitted_ranking for s in self.students}
            self.data_to_save["reasoning"] = results.get('reasoning', {})  # Add reasoning
            self.data_to_save["truthfulness"] = results.get('truthfulness', {})  # Add truthfulness
            self.data_to_save["da_trace"] = results.get('da_trace', [])

            # Compute overall truthfulness rate
            truthfulness_list = list(results.get('truthfulness', {}).values())
            if truthfulness_list:
                truthfulness_rate = sum(truthfulness_list) / len(truthfulness_list)
                self.data_to_save["truthfulness_rate"] = truthfulness_rate
                print(f"\n  Truthfulness rate: {truthfulness_rate:.1%}")

        elif self.rule.mechanism_type == "osp":
            self.data_to_save["osp_choices"] = {s.name: s.osp_choices for s in self.students}
            self.data_to_save["osp_history"] = results.get('osp_history', [])
            self.data_to_save["truthfulness"] = results.get('truthfulness', {})

            # Compute overall truthfulness rate
            truthfulness_list = list(results.get('truthfulness', {}).values())
            if truthfulness_list:
                truthfulness_rate = sum(truthfulness_list) / len(truthfulness_list)
                self.data_to_save["truthfulness_rate"] = truthfulness_rate
                print(f"\n  Overall OSP Truthfulness rate: {truthfulness_rate:.1%}")

        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Matches: {matches}")
        print(f"Utilities: {self.data_to_save['utilities']}")

    def copy_prompts(self):
        """Copy all prompt files to the prompts directory (only once)."""
        import shutil

        # Copy main template file (check if already exists)
        template_name = self.rule.special_name or f"da_{self.rule.mechanism_type}_traditional.txt"
        template_src = os.path.join(self.rule.templates_dir, template_name)
        template_dst = os.path.join(self.prompts_dir, template_name)
        if os.path.exists(template_src) and not os.path.exists(template_dst):
            shutil.copy2(template_src, template_dst)

        # Copy da_ask.txt (check if already exists)
        da_ask_src = os.path.join(prompt_dir, 'da_ask.txt')
        da_ask_dst = os.path.join(self.prompts_dir, 'da_ask.txt')
        if os.path.exists(da_ask_src) and not os.path.exists(da_ask_dst):
            shutil.copy2(da_ask_src, da_ask_dst)
            print(f"Prompts copied to: {self.prompts_dir}")

    def data_to_json(self):
        """Export results to JSON file in raw_data subdirectory."""
        # Copy prompts (will only copy once)
        self.copy_prompts()

        # Use experiment_index for filename if available, otherwise use timestring
        if self.experiment_index is not None:
            filename = f"result_{self.experiment_index}_{self.timestring}.json"
        else:
            filename = f"result_{self.timestring}.json"

        # Save to raw_data subdirectory (like auction experiments)
        filepath = save_json(self.data_to_save, filename, self.raw_data_dir)
        print(f"\nResults saved to: {filepath}")

        return filepath
