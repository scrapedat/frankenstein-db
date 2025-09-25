"""
AI Workflow Manager for FrankensteinDB

Manages the small AI model's workflow to keep it focused on either
data cleaning/structuring or real-time problem solving tasks.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Literal
from enum import Enum
import json
from dataclasses import dataclass
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

class FieldSchema(TypedDict):
    """Schema definition for a data field"""
    type: str  # Data type (string, number, datetime, etc)
    required: bool  # Whether field is required
    pattern: Optional[str]  # Regex pattern for validation
    examples: List[str]  # Example valid values
    transformation: Optional[str]  # Transform to apply

class ContentSchema(TypedDict):
    """Full content schema definition"""
    fields: Dict[str, FieldSchema]
    relationships: List[Dict[str, str]]
    validations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ValidationResult(TypedDict):
    """Result of data validation"""
    valid: bool
    errors: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    suggestions: List[str]

class PlanStep(TypedDict):
    """Type definition for a plan step"""
    action: str  # The type of action to perform
    description: str  # Description of what this step does
    params: Dict  # Parameters needed for this action
    validation: Dict  # Criteria to validate step success

from typing import TypedDict, Literal

TaskTemplate = TypedDict('TaskTemplate', {
    'name': str,
    'description': str,
    'steps': List[str],
    'validation': List[str],
    'required_context': List[str],
    'expected_output': str
})

class StructurePattern(TypedDict):
    """Pattern for data structuring tasks"""
    type: Literal['json', 'html', 'text', 'table']
    schema: Dict[str, Any]
    selectors: Dict[str, str]
    transformations: List[str]
    validation_rules: List[str]

class EditOperation(TypedDict):
    """Single edit operation"""
    type: Literal['insert', 'update', 'delete']
    target: str  # File or data identifier
    location: Dict[str, Any]  # Where to apply edit
    content: Optional[str]  # New content
    metadata: Dict[str, Any]  # Additional context

class EditTransaction(TypedDict):
    """Grouped edit operations"""
    id: str
    operations: List[EditOperation]
    dependencies: List[str]  # Other transaction IDs
    rollback: List[Dict]  # Rollback instructions
    validation: List[str]  # Validation rules

class TaskMetrics(TypedDict):
    """Metrics for task execution"""
    success_rate: float
    avg_duration: float
    error_types: Dict[str, int]
    improvement_score: float

class PatternMetrics(TypedDict):
    """Metrics for pattern effectiveness"""
    usage_count: int
    success_rate: float
    adaptations: List[Dict]
    confidence: float

class AdaptationResult(TypedDict):
    """Result of pattern adaptation"""
    original: Dict
    adapted: Dict
    improvement: float
    confidence: float

class EditResult(TypedDict):
    """Result of edit operation"""
    success: bool
    changes: List[Dict]
    conflicts: List[Dict]
    suggestions: List[str]

class TransformationRule(TypedDict):
    """Rule for data transformation"""
    pattern: str
    replacement: str
    conditions: List[str]
    priority: int

class AIWorkflowState(Enum):
    """Possible states for the DB AI workflow"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    DATA_CLEANING = "data_cleaning"
    DATA_STRUCTURING = "data_structuring"
    PROBLEM_SOLVING = "problem_solving"
    DNA_ANALYSIS = "dna_analysis"
    DNA_OPTIMIZATION = "dna_optimization"

class DatabaseTaskType(Enum):
    """Types of tasks the database AI can handle"""
    QUICK_QUERY = "quick_query"  # Fast data retrieval/update
    DNA_UPDATE = "dna_update"    # Update website DNA
    STRUCTURE = "structure"      # Data structure operations
    OPTIMIZE = "optimize"        # Optimization tasks
    ANALYZE = "analyze"         # Analysis tasks

@dataclass
class DatabaseAITask:
    """Represents a task for the database AI"""
    task_type: DatabaseTaskType
    priority: int = 1  # 1-5, higher is more important
    max_duration_ms: int = 1000  # Maximum task duration
    requires_dna: bool = False  # Whether task needs DNA data
    data: Dict = None  # Task-specific data

class AIWorkflowManager:
    """
    Manages the DB AI's workflow to maintain focus and efficiency using a DNA-aware Plan-Execute loop
    """
    
    def __init__(self, db, task_timeout: int = 300):
        self.db = db
        self.task_timeout = task_timeout
        self.current_state = AIWorkflowState.IDLE
        self.task_templates: Dict[str, TaskTemplate] = {}
        self.structure_patterns: Dict[str, StructurePattern] = {}
        self.transformation_rules: List[TransformationRule] = []
        
        # Metrics tracking
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.pattern_metrics: Dict[str, PatternMetrics] = {}
        self.adaptation_history: List[AdaptationResult] = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.confidence_threshold = 0.8
        
        self._load_templates()
        self._load_metrics()
        
    def _load_templates(self):
        """Load task templates and patterns"""
        # Data cleaning templates
        self.task_templates['clean_html'] = {
            'name': 'Clean HTML Content',
            'description': 'Remove unwanted tags and normalize structure',
            'steps': [
                'identify_content_blocks',
                'remove_ads',
                'normalize_whitespace',
                'fix_encodings'
            ],
            'validation': [
                'check_html_validity',
                'verify_content_preserved'
            ],
            'required_context': ['raw_html', 'content_rules'],
            'expected_output': 'clean_html'
        }
        
        # Structure patterns
        self.structure_patterns['article'] = {
            'type': 'html',
            'schema': {
                'title': 'string',
                'content': 'string',
                'author': 'string?',
                'date': 'datetime?'
            },
            'selectors': {
                'title': 'h1,h2.title',
                'content': 'article,div.content',
                'author': '.author,span.byline',
                'date': 'time,.date'
            },
            'transformations': [
                'clean_text',
                'normalize_dates',
                'extract_metadata'
            ],
            'validation_rules': [
                'required_fields',
                'date_format',
                'content_length'
            ]
        }
        
    async def plan_task(self, task: DatabaseAITask) -> List[PlanStep]:
        """Generate optimized execution plan using DNA patterns and templates"""
        try:
            # Get historical DNA patterns
            dna_patterns = await self._get_relevant_dna_patterns(task)
            
            # Generate base plan
            base_plan = await self._generate_base_plan(task)
            
            # Optimize using DNA patterns
            if dna_patterns and task.requires_dna:
                optimized = await self._optimize_plan_with_dna(
                    base_plan, dna_patterns, task
                )
                return optimized
            
            return base_plan
            
        except Exception as e:
            logger.error(f"Error planning task: {e}")
            return []
        """Generate an execution plan for a database task"""
        plan = []
        
        if task.requires_dna:
            # Add DNA retrieval/validation step
            plan.append({
                'action': 'prepare_dna',
                'description': 'Retrieve and validate DNA data',
                'params': {'domain': task.data.get('domain')},
                'validation': {'required': ['dna_loaded']}
            })
        
        if task.task_type == DatabaseTaskType.QUICK_QUERY:
            plan.append({
                'action': 'execute_query',
                'description': 'Execute fast database query',
                'params': task.data,
                'validation': {
                    'required': ['query_complete'],
                    'timing': {'max_ms': task.max_duration_ms}
                }
            })
            
        elif task.task_type == DatabaseTaskType.DNA_UPDATE:
            plan.extend([
                {
                    'action': 'update_dna',
                    'description': 'Update website DNA with new data',
                    'params': task.data,
                    'validation': {'required': ['dna_updated', 'metrics_updated']}
                },
                {
                    'action': 'optimize_dna',
                    'description': 'Optimize DNA structure and patterns',
                    'params': {'domain': task.data.get('domain')},
                    'validation': {'required': ['patterns_analyzed']}
                }
            ])
            
        elif task.task_type == DatabaseTaskType.STRUCTURE:
            plan.extend([
                {
                    'action': 'analyze_structure',
                    'description': 'Analyze data structure requirements',
                    'params': task.data,
                    'validation': {'required': ['structure_analyzed']}
                },
                {
                    'action': 'apply_structure',
                    'description': 'Apply optimal data structure',
                    'params': task.data,
                    'validation': {'required': ['structure_applied']}
                }
            ])
        
        return plan
    
    async def execute_step(self, step: PlanStep) -> Dict:
        """Execute a single step in the task plan"""
        start_time = time.time()
        
        try:
            if step['action'] == 'prepare_dna':
                result = await self._prepare_dna(step['params'])
            elif step['action'] == 'execute_query':
                result = await self._execute_query(step['params'])
            elif step['action'] == 'update_dna':
                result = await self._update_dna(step['params'])
            elif step['action'] == 'optimize_dna':
                result = await self._optimize_dna(step['params'])
            else:
                result = await self._execute_generic_step(step)
                
            # Validate step results
            success = self._validate_step_result(step, result)
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            await self._update_metrics(step['action'], success, duration_ms)
            
            return {'success': success, 'result': result}
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
        """
        Initialize workflow manager with DNA awareness
        
        Args:
            db: FrankensteinDB instance
            task_timeout: Maximum time (seconds) for a task before switching
        """
        self.db = db
        self.task_timeout = task_timeout
        self.current_state = AIWorkflowState.IDLE
        
        # DNA-specific tracking
        self.dna_update_queue = asyncio.Queue()
        self.dna_analysis_interval = 60  # Analyze DNA every 60s
        self.last_dna_analysis = 0
        
        # Performance tracking
        self.task_metrics: Dict[DatabaseTaskType, Dict] = {
            task_type: {
                'avg_duration_ms': 0,
                'success_rate': 1.0,
                'total_attempts': 0
            } for task_type in DatabaseTaskType
        }
        
        # Task prioritization
        self.task_queue = asyncio.PriorityQueue()  # (priority, task)
        self.current_task: Optional[Dict] = None
        self.task_start_time: Optional[float] = None
        self.task_queue: List[Dict] = []
        self.current_plan: Optional[List[Dict]] = None
        self.plan_step_index: int = 0
        
    async def _generate_pattern_adaptations(self, 
                                        pattern: Dict,
                                        metrics: TaskMetrics) -> List[Dict]:
        """Generate potential pattern adaptations"""
        adaptations = []
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(metrics)
        
        # Generate structural adaptations
        if 'structure' in pattern:
            adaptations.extend(
                self._adapt_structure(pattern, error_patterns)
            )
            
        # Generate validation adaptations
        if 'validation' in pattern:
            adaptations.extend(
                self._adapt_validation(pattern, error_patterns)
            )
            
        # Generate transformation adaptations
        if 'transformations' in pattern:
            adaptations.extend(
                self._adapt_transformations(pattern, metrics)
            )
            
        return adaptations
    
    async def _test_adaptations(self,
                             adaptations: List[Dict],
                             original: Dict) -> Optional[Dict]:
        """Test pattern adaptations and select best"""
        results = []
        
        for adaptation in adaptations:
            # Run test cases
            test_results = await self._run_adaptation_tests(
                adaptation,
                original
            )
            
            if test_results['success_rate'] > self.confidence_threshold:
                results.append({
                    'adaptation': adaptation,
                    'improvement': test_results['improvement'],
                    'confidence': test_results['confidence']
                })
                
        if not results:
            return None
            
        # Select best adaptation
        return max(results,
                  key=lambda x: x['improvement'] * x['confidence'])
    
    async def start(self):
        """Start the workflow manager"""
        logger.info("🧠 Starting AI workflow manager")
        asyncio.create_task(self._monitor_workflow())
        
    async def _monitor_workflow(self):
        """Monitor and manage AI workflow"""
        while True:
            try:
                await self._check_task_timeout()
                await self._process_next_task()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in workflow monitor: {str(e)}")
                
    async def _check_task_timeout(self):
        """Check if current task has timed out"""
        if (self.current_task and self.task_start_time and 
            time.time() - self.task_start_time > self.task_timeout):
            logger.warning(f"Task timed out: {self.current_task['type']}")
            await self.switch_to_idle()
            
    async def _process_next_task(self):
        """Process next task in queue if idle"""
        if self.current_state == AIWorkflowState.IDLE and self.task_queue:
            next_task = self.task_queue.pop(0)
            await self.start_task(next_task)
            
    async def start_task(self, task: Dict):
        """
        Start a new AI task using the Plan-Execute loop
        
        Args:
            task: Task configuration dictionary with plan
        """
        self.current_task = task
        self.current_plan = task.get('plan')
        self.plan_step_index = 0
        self.task_start_time = time.time()
        
        if not self.current_plan:
            logger.error("Task has no plan")
            await self.switch_to_idle()
            return
            
        logger.info(f"Starting task execution with {len(self.current_plan)} steps")
        await self._execute_next_step()
        
    async def _get_relevant_dna_patterns(self, task: DatabaseAITask) -> List[WebsiteDNA]:
        """Get relevant DNA patterns for task optimization"""
        try:
            patterns = []
            
            # Get patterns from database
            if task.task_type == DatabaseTaskType.DNA_UPDATE:
                patterns = await self.db.get_dna_patterns(
                    task_type=task.task_type,
                    success_rate_min=0.8,
                    limit=5
                )
            elif task.task_type == DatabaseTaskType.STRUCTURE:
                patterns = await self.db.get_structure_patterns(
                    framework_flags=task.framework_flags,
                    limit=3
                )
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting DNA patterns: {e}")
            return []
    
    async def _optimize_plan_with_dna(self,
                                    base_plan: List[PlanStep],
                                    patterns: List[WebsiteDNA],
                                    task: DatabaseAITask) -> List[PlanStep]:
        """Optimize execution plan using DNA patterns"""
        try:
            optimized = list(base_plan)
            
            # Analyze patterns for optimization opportunities
            pattern_stats = self._analyze_dna_patterns(patterns)
            
            # Reorder steps for optimal execution
            optimized = self._reorder_steps_by_success_rate(optimized, pattern_stats)
            
            # Add prefetch steps
            prefetch = self._generate_prefetch_steps(pattern_stats)
            if prefetch:
                optimized = prefetch + optimized
                
            # Add validation steps
            validation = self._generate_validation_steps(pattern_stats)
            if validation:
                optimized.extend(validation)
                
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing plan: {e}")
            return base_plan
    
    async def _generate_plan(self, task: Dict) -> Optional[List[PlanStep]]:
        """Generate execution plan with task decomposition"""
        try:
            # Convert to DatabaseAITask
            ai_task = DatabaseAITask(
                task_type=DatabaseTaskType(task.get('type', 'quick_query')),
                priority=task.get('priority', 1),
                requires_dna='dna_signature' in task,
                max_duration_ms=task.get('timeout_ms', 1000)
            )
            
            # Decompose task into subtasks
            subtasks = await self._decompose_task(ai_task)
            
            # Generate plans for subtasks
            plans = []
            for subtask in subtasks:
                template = self._find_matching_template(subtask)
                if template:
                    plan = await self._generate_from_template(subtask, template)
                    plans.extend(plan)
                else:
                    # Fallback to basic planning
                    plan = await self.plan_task(subtask)
                    plans.extend(plan)
                    
            return self._optimize_combined_plan(plans)
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return None
            
    async def _decompose_task(self, task: DatabaseAITask) -> List[DatabaseAITask]:
        """Decompose complex task into smaller subtasks"""
        subtasks = []
        
        if task.task_type == DatabaseTaskType.STRUCTURE:
            # Split structuring task by content type
            content_types = self._analyze_content_types(task)
            for ctype in content_types:
                subtask = DatabaseAITask(
                    task_type=DatabaseTaskType.STRUCTURE,
                    priority=task.priority,
                    requires_dna=task.requires_dna,
                    max_duration_ms=task.max_duration_ms // len(content_types)
                )
                subtask.content_type = ctype
                subtasks.append(subtask)
                
        elif task.task_type == DatabaseTaskType.DNA_UPDATE:
            # Split DNA update by aspect
            aspects = ['structure', 'frameworks', 'performance']
            for aspect in aspects:
                if self._needs_update(task, aspect):
                    subtask = DatabaseAITask(
                        task_type=DatabaseTaskType.DNA_UPDATE,
                        priority=task.priority,
                        requires_dna=True,
                        max_duration_ms=task.max_duration_ms // len(aspects)
                    )
                    subtask.update_aspect = aspect
                    subtasks.append(subtask)
        else:
            # Simple task doesn't need decomposition
            subtasks.append(task)
            
        return subtasks
        
    def _find_matching_template(self, task: DatabaseAITask) -> Optional[TaskTemplate]:
        """Find best matching template for task"""
        if task.task_type == DatabaseTaskType.STRUCTURE:
            content_type = getattr(task, 'content_type', None)
            if content_type in self.structure_patterns:
                return self._create_template_from_pattern(
                    self.structure_patterns[content_type]
                )
                
        template_key = f"{task.task_type.value}_{getattr(task, 'update_aspect', '')}"
        return self.task_templates.get(template_key)
        
    def _optimize_combined_plan(self, plans: List[List[PlanStep]]) -> List[PlanStep]:
        """Optimize combined execution plan"""
        # Flatten plans
        flat_plan = [step for subplan in plans for step in subplan]
        
        # Remove redundant steps
        seen_actions = set()
        optimized = []
        for step in flat_plan:
            action_key = f"{step['action']}:{step.get('target', '')}"
            if action_key not in seen_actions:
                seen_actions.add(action_key)
                optimized.append(step)
                
        # Reorder for efficiency
        return self._reorder_steps_by_success_rate(optimized, {})
        """
        Generate a plan for executing the task
        
        Args:
            task: Task configuration
            
        Returns:
            List of planned steps or None if planning failed
        """
        self.current_state = AIWorkflowState.PLANNING
        task_type = task.get('type')
        
        try:
            if task_type == 'clean':
                return await self._plan_data_cleaning(task)
            elif task_type == 'structure':
                return await self._plan_data_structuring(task)
            elif task_type == 'solve':
                return await self._plan_problem_solving(task)
            else:
                logger.error(f"Unknown task type: {task_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating plan: {str(e)}")
            return None
        finally:
            self.current_state = AIWorkflowState.IDLE
            
    def _reorder_steps_by_success_rate(self,
                                    steps: List[PlanStep],
                                    pattern_stats: Dict) -> List[PlanStep]:
        """Reorder execution steps based on historical success rates"""
        try:
            # Group steps by type
            step_groups = {}
            for step in steps:
                group = step['action'].split('_')[0]
                if group not in step_groups:
                    step_groups[group] = []
                step_groups[group].append(step)
                
            # Order groups by success rate
            ordered_groups = sorted(
                step_groups.items(),
                key=lambda x: pattern_stats.get(x[0], {}).get('success_rate', 0),
                reverse=True
            )
            
            # Rebuild step list
            ordered = []
            for _, group_steps in ordered_groups:
                ordered.extend(group_steps)
                
            return ordered
            
        except Exception as e:
            logger.error(f"Error reordering steps: {e}")
            return steps
    
    async def _execute_next_step(self):
        """
        Execute the next step in the current plan
        """
        if not self.current_plan or self.plan_step_index >= len(self.current_plan):
            await self._complete_task()
            return
            
        step = self.current_plan[self.plan_step_index]
        self.current_state = AIWorkflowState.EXECUTING
        
        try:
            logger.info(f"Executing step {self.plan_step_index + 1}/{len(self.current_plan)}: {step['description']}")
            
            if step['action'] == 'clean':
                success = await self._execute_cleaning_step(step)
            elif step['action'] == 'structure':
                success = await self._execute_structuring_step(step)
            elif step['action'] == 'solve':
                success = await self._execute_solving_step(step)
            else:
                logger.error(f"Unknown step action: {step['action']}")
                success = False
                
            if success:
                self.plan_step_index += 1
                await self._execute_next_step()
            else:
                await self._handle_step_failure(step)
                
        except Exception as e:
            logger.error(f"Error executing step: {str(e)}")
            await self._handle_step_failure(step)
            
    async def _create_edit_transaction(self, operations: List[EditOperation]) -> EditTransaction:
        """Create a new edit transaction"""
        transaction_id = str(uuid.uuid4())
        
        # Group related operations
        grouped = self._group_related_operations(operations)
        
        # Generate rollback instructions
        rollback = await self._generate_rollback_plan(grouped)
        
        # Determine dependencies
        deps = self._analyze_dependencies(grouped)
        
        return {
            'id': transaction_id,
            'operations': grouped,
            'dependencies': deps,
            'rollback': rollback,
            'validation': self._generate_validation_rules(grouped)
        }
    
    async def _execute_edit_transaction(self, transaction: EditTransaction) -> EditResult:
        """Execute an edit transaction with rollback support"""
        changes = []
        conflicts = []
        
        try:
            # Check dependencies
            if not await self._check_dependencies(transaction):
                raise ValueError("Dependencies not met")
            
            # Execute operations
            for op in transaction['operations']:
                result = await self._execute_edit_operation(op)
                if result['success']:
                    changes.append(result['changes'])
                else:
                    conflicts.append(result['conflicts'])
                    await self._rollback_transaction(transaction)
                    break
                    
            # Validate results
            validation_errors = await self._validate_changes(
                changes, transaction['validation']
            )
            if validation_errors:
                conflicts.extend(validation_errors)
                await self._rollback_transaction(transaction)
                
            return {
                'success': len(conflicts) == 0,
                'changes': changes,
                'conflicts': conflicts,
                'suggestions': self._generate_edit_suggestions(conflicts)
            }
            
        except Exception as e:
            logger.error(f"Edit transaction failed: {e}")
            await self._rollback_transaction(transaction)
            return {
                'success': False,
                'changes': [],
                'conflicts': [{'error': str(e)}],
                'suggestions': []
            }
    
    async def _handle_step_failure(self, failed_step: PlanStep):
        """
        Handle failure of a plan step
        """
        logger.error(f"Step failed: {failed_step['description']}")
        
        # Publish failure
        await self._publish_task_result(
            self.current_task,
            status='error',
            error=f"Failed at step {self.plan_step_index + 1}: {failed_step['description']}"
        )
        
        await self.switch_to_idle()
        
    async def _resolve_edit_conflicts(self, conflicts: List[Dict]) -> List[EditOperation]:
        """Generate resolution operations for edit conflicts"""
        resolutions = []
        
        for conflict in conflicts:
            if conflict['type'] == 'content_conflict':
                # Try to merge changes
                merged = await self._merge_conflicting_content(
                    conflict['current'],
                    conflict['proposed']
                )
                if merged:
                    resolutions.append({
                        'type': 'update',
                        'target': conflict['target'],
                        'location': conflict['location'],
                        'content': merged,
                        'metadata': {
                            'resolution_type': 'merge',
                            'original_conflict': conflict
                        }
                    })
            
            elif conflict['type'] == 'dependency_conflict':
                # Try to reorder operations
                reordered = self._reorder_conflicting_operations(
                    conflict['operations'],
                    conflict['dependencies']
                )
                resolutions.extend(reordered)
                
            elif conflict['type'] == 'validation_error':
                # Generate fixing operations
                fixes = await self._generate_validation_fixes(
                    conflict['error'],
                    conflict['context']
                )
                resolutions.extend(fixes)
                
        return resolutions
    
    async def _complete_task(self):
        """
        Handle successful task completion
        """
        logger.info("Task completed successfully")
        
        await self._publish_task_result(
            self.current_task,
            status='success',
            result={
                'steps_completed': self.plan_step_index,
                'total_steps': len(self.current_plan) if self.current_plan else 0,
                'execution_time': time.time() - self.task_start_time
            }
        )
        
        await self.switch_to_idle()
            
    async def _start_data_task(self, task: Dict):
        """Start data cleaning or structuring task"""
        self.current_task = task
        self.task_start_time = time.time()
        
        if task['type'] == 'clean':
            self.current_state = AIWorkflowState.DATA_CLEANING
            await self._clean_data(task['data'])
        else:
            self.current_state = AIWorkflowState.DATA_STRUCTURING
            await self._structure_data(task['data'])
            
    async def _start_problem_solving(self, task: Dict):
        """Start problem solving task"""
        self.current_task = task
        self.task_start_time = time.time()
        self.current_state = AIWorkflowState.PROBLEM_SOLVING
        await self._solve_problem(task['data'])
        
    async def queue_task(self, task: Dict) -> bool:
        """
        Queue a new task if compatible with current state
        
        Args:
            task: Task configuration
            
        Returns:
            Whether task was queued successfully
        """
        # Don't queue if similar task is already running
        if (self.current_task and 
            self.current_task['type'] == task['type']):
            return False
            
        # Don't queue if too many pending tasks
        if len(self.task_queue) >= 10:
            return False
            
        # Generate plan for task
        plan = await self._generate_plan(task)
        if plan:
            task['plan'] = plan
            self.task_queue.append(task)
            return True
            
        return False
        
    async def switch_to_idle(self):
        """Switch to idle state"""
        if self.current_task:
            await self._publish_task_result(
                self.current_task,
                status='timeout'
            )
            
        self.current_state = AIWorkflowState.IDLE
        self.current_task = None
        self.task_start_time = None
        
    async def _clean_data(self, data: Dict):
        """
        Clean and normalize website data
        
        Args:
            data: Website data to clean
        """
        try:
            # Apply cleaning rules
            cleaned = await self._apply_cleaning_rules(data)
            
            # Store cleaned data
            if cleaned:
                await self.db.store_website_snapshot(
                    url=data['url'],
                    html_content=cleaned['html'],
                    structure_fingerprint=cleaned['structure'],
                    keywords=cleaned.get('keywords')
                )
                
            await self._publish_task_result(
                self.current_task,
                status='success',
                result=cleaned
            )
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            await self._publish_task_result(
                self.current_task,
                status='error',
                error=str(e)
            )
            
        await self.switch_to_idle()
        
    async def _structure_data(self, data: Dict):
        """
        Structure and organize website data
        
        Args:
            data: Website data to structure
        """
        try:
            # Extract and structure data
            structured = await self._apply_structuring_rules(data)
            
            # Store structured insights
            if structured:
                await self.db.store_ai_knowledge(
                    f"structure:{data['url']}",
                    structured,
                    ttl=86400
                )
                
            await self._publish_task_result(
                self.current_task,
                status='success',
                result=structured
            )
            
        except Exception as e:
            logger.error(f"Error structuring data: {str(e)}")
            await self._publish_task_result(
                self.current_task,
                status='error',
                error=str(e)
            )
            
        await self.switch_to_idle()
        
    async def _solve_problem(self, data: Dict):
        """
        Solve a specific scraping problem
        
        Args:
            data: Problem details
        """
        try:
            # Analyze problem and generate solution
            solution = await self._generate_problem_solution(data)
            
            # Store solution pattern
            if solution:
                await self.db.store_scraping_logic(
                    data['pattern'],
                    solution,
                    ttl=604800  # 1 week
                )
                
            await self._publish_task_result(
                self.current_task,
                status='success',
                result=solution
            )
            
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}")
            await self._publish_task_result(
                self.current_task,
                status='error',
                error=str(e)
            )
            
        await self.switch_to_idle()
        
    async def _create_template_from_pattern(self, pattern: StructurePattern) -> TaskTemplate:
        """Create task template from structure pattern"""
        steps = []
        validation = []
        
        # Add extraction steps
        for field, selector in pattern['selectors'].items():
            steps.append(f"extract_{field}")
            
        # Add transformations
        steps.extend(pattern['transformations'])
        
        # Add validation rules
        validation.extend(pattern['validation_rules'])
        
        return {
            'name': f"Structure {pattern['type']} content",
            'description': f"Extract and structure {pattern['type']} content using schema",
            'steps': steps,
            'validation': validation,
            'required_context': ['content', 'schema'],
            'expected_output': 'structured_data'
        }
    
    async def _plan_data_cleaning(self, task: Dict) -> List[PlanStep]:
        """Plan data cleaning using templates and patterns"""
        template = self.task_templates.get('clean_html')
        if not template:
            return []
            
        plan = []
        for step in template['steps']:
            plan.append({
                'action': step,
                'description': f"Execute {step} on content",
                'params': {
                    'rules': task.get('cleaning_rules', {}),
                    'content_type': task.get('content_type', 'html')
                },
                'validation': {
                    'required': True,
                    'rules': template['validation']
                }
            })
            
        return plan
        """Generate plan for data cleaning task"""
        return [
            {
                'action': 'clean',
                'description': 'Clean HTML content',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['html', 'structure']}
            },
            {
                'action': 'clean',
                'description': 'Extract and clean metadata',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['keywords', 'description']}
            }
        ]
        
    async def _infer_schema(self, samples: List[Dict]) -> ContentSchema:
        """Infer data schema from samples"""
        fields: Dict[str, FieldSchema] = {}
        relationships = []
        validations = []
        
        # Analyze each sample
        for sample in samples:
            for key, value in sample.items():
                if key not in fields:
                    field_schema = self._infer_field_schema(key, value)
                    fields[key] = field_schema
                else:
                    # Update existing field schema
                    self._update_field_schema(fields[key], value)
                    
            # Look for relationships
            found_rels = self._detect_relationships(sample)
            for rel in found_rels:
                if rel not in relationships:
                    relationships.append(rel)
                    
        # Generate validations
        validations = self._generate_validations(fields, relationships)
        
        return {
            'fields': fields,
            'relationships': relationships,
            'validations': validations,
            'metadata': {
                'sample_size': len(samples),
                'generated': time.time()
            }
        }
    
    def _infer_field_schema(self, key: str, value: Any) -> FieldSchema:
        """Infer schema for a single field"""
        field_type = self._detect_type(value)
        pattern = self._detect_pattern(str(value)) if field_type == 'string' else None
        
        return {
            'type': field_type,
            'required': True,  # Start strict, relax if needed
            'pattern': pattern,
            'examples': [str(value)],
            'transformation': self._suggest_transformation(key, value)
        }
    
    async def _plan_data_structuring(self, task: Dict) -> List[PlanStep]:
        """Plan data structuring with schema inference"""
        plan = []
        
        # Analyze input
        plan.append({
            'action': 'analyze_input',
            'description': 'Analyze input data structure',
            'params': {
                'sample_size': 10
            },
            'validation': {
                'required': True
            }
        })
        
        # Infer schema
        plan.append({
            'action': 'infer_schema',
            'description': 'Infer data schema from samples',
            'params': {
                'min_confidence': 0.8
            },
            'validation': {
                'required': True,
                'rules': ['schema_completeness', 'type_consistency']
            }
        })
        
        # Transform data
        plan.append({
            'action': 'transform_data',
            'description': 'Apply schema transformations',
            'params': {
                'strict_mode': True
            },
            'validation': {
                'required': True,
                'rules': ['data_consistency', 'relationship_validity']
            }
        })
        
        return plan
        """Generate plan for data structuring task"""
        return [
            {
                'action': 'structure',
                'description': 'Analyze document structure',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['elements', 'hierarchy']}
            },
            {
                'action': 'structure',
                'description': 'Extract structured data',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['extracted_data']}
            }
        ]
        
    async def _plan_problem_solving(self, task: Dict) -> List[PlanStep]:
        """Generate plan for problem solving task"""
        return [
            {
                'action': 'solve',
                'description': 'Analyze problem context',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['context', 'constraints']}
            },
            {
                'action': 'solve',
                'description': 'Generate solution strategy',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['strategy', 'steps']}
            },
            {
                'action': 'solve',
                'description': 'Validate solution',
                'params': {'data': task['data']},
                'validation': {'required_fields': ['validation_results']}
            }
        ]
        
    async def _execute_cleaning_step(self, step: PlanStep) -> bool:
        """Execute a cleaning step"""
        cleaned = await self._apply_cleaning_rules(step['params']['data'])
        if not cleaned:
            return False
            
        # Validate step results
        return all(field in cleaned for field in step['validation']['required_fields'])
        
    async def _execute_structuring_step(self, step: PlanStep) -> bool:
        """Execute a structuring step"""
        structured = await self._apply_structuring_rules(step['params']['data'])
        if not structured:
            return False
            
        # Validate step results
        return all(field in structured for field in step['validation']['required_fields'])
        
    async def _execute_solving_step(self, step: PlanStep) -> bool:
        """Execute a problem solving step"""
        solution = await self._generate_problem_solution(step['params']['data'])
        if not solution:
            return False
            
        # Validate step results
        return all(field in solution for field in step['validation']['required_fields'])
        
    async def _apply_cleaning_rules(self, data: Dict) -> Optional[Dict]:
        """Apply data cleaning rules"""
        # TODO: Implement small model cleaning rules
        return None
        
    async def _apply_structuring_rules(self, data: Dict) -> Optional[Dict]:
        """Apply data structuring rules"""
        # TODO: Implement small model structuring rules
        return None
        
    async def _generate_problem_solution(self, data: Dict) -> Optional[Dict]:
        """Generate solution for scraping problem"""
        # TODO: Implement small model problem solving
        return None
        
    async def _publish_task_result(self, task: Dict, status: str, 
                                result: Optional[Dict] = None,
                                error: Optional[str] = None):
        """Publish task result via MQTT"""
        if not task:
            return
            
        await self.db.mqtt.publish(
            'db/ai/task_result',
            {
                'task_id': task.get('id'),
                'type': task.get('type'),
                'status': status,
                'result': result,
                'error': error,
                'timestamp': time.time()
            }
        )