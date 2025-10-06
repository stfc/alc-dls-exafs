"""Comprehensive tests for the pipeline module.

Tests cover the three-stage EXAFS processing architecture:
1. Input generation (InputGenerator)
2. FEFF execution (FeffExecutor)
3. Result processing (ResultProcessor)
4. Unified processing (PipelineProcessor)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from larch import Group

from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.pipeline import (
    FeffBatch,
    FeffExecutor,
    FeffTask,
    InputGenerator,
    PipelineProcessor,
    ResultProcessor,
)

# ============================================================================
# Test Data and Fixtures
# ============================================================================


@pytest.fixture
def simple_atoms():
    """Create a simple test Atoms object."""
    return Atoms("Au2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10, 10, 10])


@pytest.fixture
def complex_atoms():
    """Create a more complex test Atoms object."""
    positions = [
        [0, 0, 0],  # Au absorber
        [2.5, 0, 0],  # Au neighbor
        [0, 2.5, 0],  # Ag neighbor
        [2.5, 2.5, 0],  # Ag neighbor
    ]
    return Atoms("Au2Ag2", positions=positions, cell=[10, 10, 10])


@pytest.fixture
def trajectory_atoms(simple_atoms):
    """Create a trajectory of test Atoms objects."""
    structures = []
    for _i in range(3):
        atoms = simple_atoms.copy()
        # Slightly perturb positions for each frame
        atoms.positions += np.random.random((len(atoms), 3)) * 0.1
        structures.append(atoms)
    return structures


@pytest.fixture
def test_config():
    """Create a test FeffConfig object."""
    return FeffConfig(
        radius=6.0,
        edge="K",
        kweight=2,
        kmin=2.0,
        kmax=12.0,
        dk=1.0,
        window="hanning",
        cleanup_feff_files=True,
        parallel=True,
        n_workers=2,
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_feff_task(temp_output_dir):
    """Create a sample FeffTask for testing."""
    input_file = temp_output_dir / "feff.inp"
    input_file.write_text("CONTROL 1 1 1 1 1 1\nPRINT 1 0 0 0 0 3\n")
    return FeffTask(
        input_file=input_file,
        site_index=0,
        frame_index=0,
        absorber_element="Au",
    )


@pytest.fixture
def sample_batch(sample_feff_task, test_config, temp_output_dir):
    """Create a sample FeffBatch for testing."""
    return FeffBatch(
        tasks=[sample_feff_task],
        output_dir=temp_output_dir,
        config=test_config,
    )


@pytest.fixture
def mock_larch_group():
    """Create a mock Larch Group for testing."""
    group = Group()
    group.k = np.linspace(0, 15, 100)
    group.chi = np.sin(group.k) * np.exp(-group.k / 10)
    group.r = np.linspace(0, 6, 100)
    group.chir_mag = np.abs(np.fft.fft(group.chi))[:100]
    group.chir_re = np.real(np.fft.fft(group.chi))[:100]
    group.chir_im = np.imag(np.fft.fft(group.chi))[:100]
    return group


# ============================================================================
# Test FeffTask
# ============================================================================


class TestFeffTask:
    """Test the FeffTask dataclass."""

    def test_feff_task_creation(self, temp_output_dir):
        """Test basic FeffTask creation."""
        input_file = temp_output_dir / "feff.inp"
        task = FeffTask(
            input_file=input_file,
            site_index=1,
            frame_index=2,
            absorber_element="Au",
        )

        assert task.input_file == input_file
        assert task.site_index == 1
        assert task.frame_index == 2
        assert task.absorber_element == "Au"

    def test_feff_task_defaults(self, temp_output_dir):
        """Test FeffTask with default values."""
        input_file = temp_output_dir / "feff.inp"
        task = FeffTask(input_file=input_file, site_index=0)

        assert task.frame_index == 0
        assert task.absorber_element == ""

    def test_feff_dir_property(self, temp_output_dir):
        """Test the feff_dir property."""
        input_file = temp_output_dir / "subdir" / "feff.inp"
        task = FeffTask(input_file=input_file, site_index=0)

        assert task.feff_dir == input_file.parent

    def test_task_id_property(self, temp_output_dir):
        """Test the task_id property."""
        input_file = temp_output_dir / "feff.inp"
        task = FeffTask(
            input_file=input_file,
            site_index=5,
            frame_index=3,
        )

        assert task.task_id == "frame_0003_site_0005"


# ============================================================================
# Test FeffBatch
# ============================================================================


class TestFeffBatch:
    """Test the FeffBatch dataclass."""

    def test_feff_batch_creation(self, sample_feff_task, test_config, temp_output_dir):
        """Test basic FeffBatch creation."""
        batch = FeffBatch(
            tasks=[sample_feff_task],
            output_dir=temp_output_dir,
            config=test_config,
        )

        assert len(batch.tasks) == 1
        assert batch.output_dir == temp_output_dir
        assert batch.config == test_config

    def test_get_tasks_by_frame(self, test_config, temp_output_dir):
        """Test grouping tasks by frame index."""
        tasks = []
        for frame_idx in range(2):
            for site_idx in range(3):
                input_file = (
                    temp_output_dir
                    / f"frame_{frame_idx}"
                    / f"site_{site_idx}"
                    / "feff.inp"
                )
                task = FeffTask(
                    input_file=input_file,
                    site_index=site_idx,
                    frame_index=frame_idx,
                )
                tasks.append(task)

        batch = FeffBatch(tasks=tasks, output_dir=temp_output_dir, config=test_config)
        frames = batch.get_tasks_by_frame()

        assert len(frames) == 2
        assert len(frames[0]) == 3
        assert len(frames[1]) == 3

        # Check that tasks are grouped correctly
        for frame_idx, frame_tasks in frames.items():
            for task in frame_tasks:
                assert task.frame_index == frame_idx

    def test_get_tasks_by_site(self, test_config, temp_output_dir):
        """Test grouping tasks by site index."""
        tasks = []
        for frame_idx in range(3):
            for site_idx in range(2):
                input_file = (
                    temp_output_dir
                    / f"frame_{frame_idx}"
                    / f"site_{site_idx}"
                    / "feff.inp"
                )
                task = FeffTask(
                    input_file=input_file,
                    site_index=site_idx,
                    frame_index=frame_idx,
                )
                tasks.append(task)

        batch = FeffBatch(tasks=tasks, output_dir=temp_output_dir, config=test_config)
        sites = batch.get_tasks_by_site()

        assert len(sites) == 2
        assert len(sites[0]) == 3
        assert len(sites[1]) == 3

        # Check that tasks are grouped correctly
        for site_idx, site_tasks in sites.items():
            for task in site_tasks:
                assert task.site_index == site_idx


# ============================================================================
# Test InputGenerator
# ============================================================================


class TestInputGenerator:
    """Test the InputGenerator class."""

    def test_input_generator_creation(self, test_config):
        """Test InputGenerator initialization."""
        generator = InputGenerator(test_config)
        assert generator.config == test_config
        assert generator.logger is not None

    @patch("larch_cli_wrapper.feff_utils.normalize_absorbers")
    @patch("larch_cli_wrapper.feff_utils.generate_multi_site_feff_inputs")
    def test_generate_single_site_inputs(
        self,
        mock_generate,
        mock_normalize,
        test_config,
        simple_atoms,
        temp_output_dir,
    ):
        """Test generating inputs for a single site."""
        # Setup mocks
        mock_normalize.return_value = [0]
        mock_input_files = [temp_output_dir / "site_0000" / "feff.inp"]
        mock_generate.return_value = mock_input_files

        generator = InputGenerator(test_config)
        batch = generator.generate_single_site_inputs(
            structure=simple_atoms,
            absorber="Au",
            output_dir=temp_output_dir,
            frame_index=1,
        )

        # Verify mocks called correctly
        mock_normalize.assert_called_once_with(simple_atoms, "Au")
        mock_generate.assert_called_once_with(
            atoms=simple_atoms,
            absorber_indices=[0],
            base_output_dir=temp_output_dir,
            config=test_config,
        )

        # Verify batch creation
        assert isinstance(batch, FeffBatch)
        assert len(batch.tasks) == 1
        assert batch.tasks[0].site_index == 0
        assert batch.tasks[0].frame_index == 1
        assert batch.tasks[0].absorber_element == "Au"

    @patch("larch_cli_wrapper.feff_utils.normalize_absorbers")
    @patch("larch_cli_wrapper.feff_utils.generate_multi_site_feff_inputs")
    def test_generate_single_site_inputs_multiple_sites(
        self,
        mock_generate,
        mock_normalize,
        test_config,
        complex_atoms,
        temp_output_dir,
    ):
        """Test generating inputs for multiple sites."""
        # Setup mocks
        mock_normalize.return_value = [0, 1]
        mock_input_files = [
            temp_output_dir / "site_0000" / "feff.inp",
            temp_output_dir / "site_0001" / "feff.inp",
        ]
        mock_generate.return_value = mock_input_files

        generator = InputGenerator(test_config)
        batch = generator.generate_single_site_inputs(
            structure=complex_atoms,
            absorber=[0, 1],
            output_dir=temp_output_dir,
        )

        # Verify batch has correct number of tasks
        assert len(batch.tasks) == 2
        assert batch.tasks[0].site_index == 0
        assert batch.tasks[1].site_index == 1
        assert all(task.frame_index == 0 for task in batch.tasks)

    def test_generate_trajectory_inputs(
        self,
        test_config,
        trajectory_atoms,
        temp_output_dir,
    ):
        """Test generating inputs for trajectory."""
        generator = InputGenerator(test_config)

        # Mock the single site generation
        with patch.object(generator, "generate_single_site_inputs") as mock_single:
            # Create mock batches for each frame
            mock_batches = []
            for i in range(len(trajectory_atoms)):
                task = FeffTask(
                    input_file=temp_output_dir
                    / f"frame_{i:04d}"
                    / "site_0000"
                    / "feff.inp",
                    site_index=0,
                    frame_index=i,
                )
                mock_batch = FeffBatch(
                    tasks=[task], output_dir=temp_output_dir, config=test_config
                )
                mock_batches.append(mock_batch)

            mock_single.side_effect = mock_batches

            batch = generator.generate_trajectory_inputs(
                structures=trajectory_atoms,
                absorber="Au",
                output_dir=temp_output_dir,
            )

            # Verify single site generation called for each frame
            assert mock_single.call_count == len(trajectory_atoms)

            # Verify final batch
            assert len(batch.tasks) == len(trajectory_atoms)
            for i, task in enumerate(batch.tasks):
                assert task.frame_index == i


# ============================================================================
# Test FeffExecutor
# ============================================================================


class TestFeffExecutor:
    """Test the FeffExecutor class."""

    def test_feff_executor_creation(self, temp_output_dir):
        """Test FeffExecutor initialization."""
        executor = FeffExecutor(
            max_workers=4,
            cache_dir=temp_output_dir / "cache",
            force_recalculate=True,
        )

        assert executor.max_workers == 4
        assert executor.cache_dir == temp_output_dir / "cache"
        assert executor.force_recalculate is True
        assert executor.logger is not None

    def test_feff_executor_cache_creation(self, temp_output_dir):
        """Test that cache directory is created."""
        cache_dir = temp_output_dir / "cache"
        assert not cache_dir.exists()

        FeffExecutor(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_get_feff_input_hash(self, temp_output_dir):
        """Test FEFF input file hashing."""
        executor = FeffExecutor()

        # Create test input file
        input_file = temp_output_dir / "feff.inp"
        content = "CONTROL 1 1 1 1 1 1\nPRINT 1 0 0 0 0 3\nEDGE K\n"
        input_file.write_text(content)

        hash1 = executor._get_feff_input_hash(input_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same content should give same hash
        hash2 = executor._get_feff_input_hash(input_file)
        assert hash1 == hash2

        # Different content should give different hash
        input_file.write_text(content + "\nEXCHANGE 0 0.0 0.0 2\n")
        hash3 = executor._get_feff_input_hash(input_file)
        assert hash1 != hash3

    def test_get_feff_input_hash_ignores_comments(self, temp_output_dir):
        """Test that hash ignores comment lines."""
        executor = FeffExecutor()

        # Create input files with same parameters but different comments
        input_file1 = temp_output_dir / "feff1.inp"
        content1 = "CONTROL 1 1 1 1 1 1\n* This is a comment\nPRINT 1 0 0 0 0 3\n"
        input_file1.write_text(content1)

        input_file2 = temp_output_dir / "feff2.inp"
        content2 = "CONTROL 1 1 1 1 1 1\n* Different comment\nPRINT 1 0 0 0 0 3\n"
        input_file2.write_text(content2)

        hash1 = executor._get_feff_input_hash(input_file1)
        hash2 = executor._get_feff_input_hash(input_file2)
        assert hash1 == hash2

    @patch("larch_cli_wrapper.cache_utils.load_from_cache")
    def test_load_cached_result_with_cache(self, mock_load, temp_output_dir):
        """Test loading cached results when cache is enabled."""
        executor = FeffExecutor(cache_dir=temp_output_dir / "cache")
        mock_data = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        mock_load.return_value = mock_data

        result = executor._load_cached_result("test_key")
        assert result == mock_data
        mock_load.assert_called_once_with("test_key", executor.cache_dir, False)

    def test_load_cached_result_no_cache(self):
        """Test loading cached results when cache is disabled."""
        executor = FeffExecutor(cache_dir=None)
        result = executor._load_cached_result("test_key")
        assert result is None

    def test_load_cached_result_force_recalculate(self, temp_output_dir):
        """Test that force_recalculate bypasses cache."""
        executor = FeffExecutor(
            cache_dir=temp_output_dir / "cache",
            force_recalculate=True,
        )
        result = executor._load_cached_result("test_key")
        assert result is None

    @patch("larch_cli_wrapper.cache_utils.save_to_cache")
    def test_save_to_cache_with_cache(self, mock_save, temp_output_dir):
        """Test saving to cache when cache is enabled."""
        executor = FeffExecutor(cache_dir=temp_output_dir / "cache")
        chi = np.array([1, 2, 3])
        k = np.array([4, 5, 6])

        executor._save_to_cache("test_key", chi, k)
        mock_save.assert_called_once_with("test_key", chi, k, executor.cache_dir)

    def test_save_to_cache_no_cache(self):
        """Test saving to cache when cache is disabled."""
        executor = FeffExecutor(cache_dir=None)
        # Should not raise any exceptions
        executor._save_to_cache("test_key", np.array([1, 2, 3]), np.array([4, 5, 6]))

    @patch("larch_cli_wrapper.feff_utils.run_multi_site_feff_calculations")
    def test_execute_batch_no_cache(
        self,
        mock_run_feff,
        sample_batch,
        temp_output_dir,
    ):
        """Test executing batch without caching."""
        # Setup mocks
        mock_run_feff.return_value = [(temp_output_dir, True)]

        executor = FeffExecutor(cache_dir=None)
        results = executor.execute_batch(sample_batch, parallel=False)

        # Verify FEFF execution called
        mock_run_feff.assert_called_once()
        call_args = mock_run_feff.call_args
        assert len(call_args[1]["input_files"]) == 1
        assert call_args[1]["parallel"] is False

        # Verify results
        assert len(results) == 1
        task_id = list(results.keys())[0]
        assert results[task_id] is True

    @patch("larch_cli_wrapper.feff_utils.run_multi_site_feff_calculations")
    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    def test_execute_batch_with_caching(
        self,
        mock_read_output,
        mock_run_feff,
        sample_batch,
        temp_output_dir,
    ):
        """Test executing batch with caching enabled."""
        # Setup mocks
        mock_run_feff.return_value = [(temp_output_dir, True)]
        mock_read_output.return_value = (np.array([1, 2, 3]), np.array([4, 5, 6]))

        executor = FeffExecutor(cache_dir=temp_output_dir / "cache")

        with patch.object(executor, "_save_to_cache") as mock_save:
            results = executor.execute_batch(sample_batch, parallel=True)

            # Verify caching called
            mock_save.assert_called_once()

        assert len(results) == 1

    def test_execute_batch_with_cached_hit(self, sample_batch, temp_output_dir):
        """Test executing batch with cache hit."""
        executor = FeffExecutor(cache_dir=temp_output_dir / "cache")

        # Mock cache hit
        cached_data = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        with patch.object(executor, "_load_cached_result", return_value=cached_data):
            with patch(
                "larch_cli_wrapper.feff_utils.run_multi_site_feff_calculations"
            ) as mock_run:
                results = executor.execute_batch(sample_batch)

                # Verify FEFF execution NOT called (cache hit)
                mock_run.assert_not_called()

                # Verify results
                assert len(results) == 1
                task_id = list(results.keys())[0]
                assert results[task_id] is True

                # Verify chi.dat file was created
                chi_file = sample_batch.tasks[0].feff_dir / "chi.dat"
                assert chi_file.exists()


# ============================================================================
# Test ResultProcessor
# ============================================================================


class TestResultProcessor:
    """Test the ResultProcessor class."""

    def test_result_processor_creation(self, test_config):
        """Test ResultProcessor initialization."""
        processor = ResultProcessor(test_config)
        assert processor.config == test_config
        assert processor.logger is not None

    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    @patch("larch.xafs.xftf")
    def test_load_successful_results(
        self,
        mock_xftf,
        mock_read_output,
        test_config,
        sample_batch,
    ):
        """Test loading successful EXAFS results."""
        # Setup mocks
        chi = np.array([1, 2, 3])
        k = np.array([4, 5, 6])
        mock_read_output.return_value = (chi, k)

        processor = ResultProcessor(test_config)
        task_results = {sample_batch.tasks[0].task_id: True}

        groups = processor.load_successful_results(sample_batch, task_results)

        # Verify results
        assert len(groups) == 1
        task_id = sample_batch.tasks[0].task_id
        group = groups[task_id]

        assert hasattr(group, "k")
        assert hasattr(group, "chi")
        assert hasattr(group, "site_idx")
        assert hasattr(group, "frame_idx")
        assert hasattr(group, "absorber_element")
        assert hasattr(group, "task_id")

        # Verify Fourier transform called
        mock_xftf.assert_called_once()

    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    def test_load_successful_results_failed_task(
        self,
        mock_read_output,
        test_config,
        sample_batch,
    ):
        """Test loading results when task failed."""
        processor = ResultProcessor(test_config)
        task_results = {sample_batch.tasks[0].task_id: False}

        groups = processor.load_successful_results(sample_batch, task_results)

        # Should return empty dict for failed tasks
        assert len(groups) == 0
        mock_read_output.assert_not_called()

    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    def test_load_successful_results_read_exception(
        self,
        mock_read_output,
        test_config,
        sample_batch,
    ):
        """Test handling of read exceptions."""
        # Setup mock to raise a realistic exception that read_feff_output can raise
        mock_read_output.side_effect = ValueError("Read failed")

        processor = ResultProcessor(test_config)
        task_results = {sample_batch.tasks[0].task_id: True}

        groups = processor.load_successful_results(sample_batch, task_results)

        # Should handle exception gracefully
        assert len(groups) == 0

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_frame_averages_single_group(
        self,
        mock_create_avg,
        test_config,
        mock_larch_group,
        sample_batch,
    ):
        """Test creating frame averages with single group."""
        processor = ResultProcessor(test_config)
        task_id = sample_batch.tasks[0].task_id
        groups = {task_id: mock_larch_group}

        frame_averages = processor.create_frame_averages(groups, sample_batch)

        # Should not call averaging for single group
        mock_create_avg.assert_not_called()

        # Verify result
        assert len(frame_averages) == 1
        assert 0 in frame_averages
        avg_group = frame_averages[0]
        assert hasattr(avg_group, "frame_idx")
        assert hasattr(avg_group, "is_average")
        assert hasattr(avg_group, "average_type")
        assert hasattr(avg_group, "n_components")

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_frame_averages_multiple_groups(
        self,
        mock_create_avg,
        test_config,
        mock_larch_group,
        temp_output_dir,
    ):
        """Test creating frame averages with multiple groups."""
        # Create batch with multiple tasks in same frame
        tasks = []
        for i in range(3):
            input_file = temp_output_dir / f"site_{i:04d}" / "feff.inp"
            task = FeffTask(
                input_file=input_file,
                site_index=i,
                frame_index=0,
            )
            tasks.append(task)

        batch = FeffBatch(tasks=tasks, output_dir=temp_output_dir, config=test_config)

        # Create groups for all tasks
        groups = {task.task_id: mock_larch_group for task in tasks}

        # Mock averaging
        averaged_group = Group()
        averaged_group.k = mock_larch_group.k
        averaged_group.chi = mock_larch_group.chi
        mock_create_avg.return_value = averaged_group

        processor = ResultProcessor(test_config)
        frame_averages = processor.create_frame_averages(groups, batch)

        # Verify averaging called
        mock_create_avg.assert_called_once()

        # Verify result
        assert len(frame_averages) == 1
        assert 0 in frame_averages

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_site_averages(
        self,
        mock_create_avg,
        test_config,
        mock_larch_group,
        temp_output_dir,
    ):
        """Test creating site averages."""
        # Create batch with same site across multiple frames
        tasks = []
        for frame in range(3):
            input_file = (
                temp_output_dir / f"frame_{frame:04d}" / "site_0000" / "feff.inp"
            )
            task = FeffTask(
                input_file=input_file,
                site_index=0,
                frame_index=frame,
            )
            tasks.append(task)

        batch = FeffBatch(tasks=tasks, output_dir=temp_output_dir, config=test_config)
        groups = {task.task_id: mock_larch_group for task in tasks}

        # Mock averaging
        averaged_group = Group()
        averaged_group.k = mock_larch_group.k
        averaged_group.chi = mock_larch_group.chi
        mock_create_avg.return_value = averaged_group

        processor = ResultProcessor(test_config)
        site_averages = processor.create_site_averages(groups, batch)

        # Verify result
        assert len(site_averages) == 1
        assert 0 in site_averages
        avg_group = site_averages[0]
        assert hasattr(avg_group, "site_idx")
        assert avg_group.average_type == "site"

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_overall_average_empty(self, mock_create_avg, test_config):
        """Test creating overall average with empty list."""
        processor = ResultProcessor(test_config)
        result = processor.create_overall_average([])

        assert result is None
        mock_create_avg.assert_not_called()

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_overall_average_single(
        self,
        mock_create_avg,
        test_config,
        mock_larch_group,
    ):
        """Test creating overall average with single group."""
        processor = ResultProcessor(test_config)
        result = processor.create_overall_average([mock_larch_group])

        # Should not call averaging for single group
        mock_create_avg.assert_not_called()
        assert result == mock_larch_group
        assert hasattr(result, "is_average")
        assert result.average_type == "overall"

    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_create_overall_average_multiple(
        self,
        mock_create_avg,
        test_config,
        mock_larch_group,
    ):
        """Test creating overall average with multiple groups."""
        groups = [mock_larch_group] * 3
        averaged_group = Group()
        averaged_group.k = mock_larch_group.k
        averaged_group.chi = mock_larch_group.chi
        mock_create_avg.return_value = averaged_group

        processor = ResultProcessor(test_config)
        result = processor.create_overall_average(groups)

        # Verify averaging called
        mock_create_avg.assert_called_once_with(groups, test_config.fourier_params)
        assert result == averaged_group
        assert hasattr(result, "is_average")
        assert result.average_type == "overall"


# ============================================================================
# Test PipelineProcessor
# ============================================================================


class TestPipelineProcessor:
    """Test the PipelineProcessor class."""

    def test_pipeline_processor_creation(self, test_config, temp_output_dir):
        """Test PipelineProcessor initialization."""
        processor = PipelineProcessor(
            config=test_config,
            max_workers=4,
            cache_dir=temp_output_dir / "cache",
            force_recalculate=True,
        )

        assert processor.config == test_config
        assert isinstance(processor.input_generator, InputGenerator)
        assert isinstance(processor.feff_executor, FeffExecutor)
        assert isinstance(processor.result_processor, ResultProcessor)
        assert processor.logger is not None

    def test_process_trajectory(
        self,
        test_config,
        trajectory_atoms,
        temp_output_dir,
        mock_larch_group,
    ):
        """Test processing a trajectory."""
        processor = PipelineProcessor(test_config)

        # Mock all the stages
        with patch.object(
            processor.input_generator, "generate_trajectory_inputs"
        ) as mock_gen:
            with patch.object(processor.feff_executor, "execute_batch") as mock_exec:
                with patch.object(
                    processor.result_processor, "load_successful_results"
                ) as mock_load:
                    with patch.object(
                        processor.result_processor, "create_frame_averages"
                    ) as mock_frame_avg:
                        with patch.object(
                            processor.result_processor, "create_site_averages"
                        ) as mock_site_avg:
                            with patch.object(
                                processor.result_processor, "create_overall_average"
                            ) as mock_overall_avg:
                                # Setup mocks
                                mock_batch = MagicMock()
                                mock_gen.return_value = mock_batch

                                mock_exec.return_value = {
                                    "task_1": True,
                                    "task_2": True,
                                }

                                mock_groups = {
                                    "task_1": mock_larch_group,
                                    "task_2": mock_larch_group,
                                }
                                mock_load.return_value = mock_groups

                                mock_frame_averages = {
                                    0: mock_larch_group,
                                    1: mock_larch_group,
                                }
                                mock_frame_avg.return_value = mock_frame_averages

                                mock_site_averages = {0: mock_larch_group}
                                mock_site_avg.return_value = mock_site_averages

                                mock_overall_avg.return_value = mock_larch_group

                                # Execute
                                result = processor.process_trajectory(
                                    structures=trajectory_atoms,
                                    absorber="Au",
                                    output_dir=temp_output_dir,
                                    parallel=True,
                                )

                                overall_avg, frame_avg, site_avg, individual = result

                                # Verify all stages called
                                mock_gen.assert_called_once()
                                mock_exec.assert_called_once()
                                mock_load.assert_called_once()
                                mock_frame_avg.assert_called_once()
                                mock_site_avg.assert_called_once()
                                mock_overall_avg.assert_called_once()

                                # Verify results
                                assert overall_avg == mock_larch_group
                                assert len(frame_avg) == 2
                                assert len(site_avg) == 1
                                assert len(individual) == 2

    def test_get_cache_info_no_cache(self, test_config):
        """Test getting cache info when cache is disabled."""
        processor = PipelineProcessor(test_config, cache_dir=None)
        info = processor.get_cache_info()

        assert info["enabled"] is False
        assert info["cache_dir"] is None
        assert info["files"] == 0
        assert info["size_mb"] == 0.0

    def test_get_cache_info_with_cache(self, test_config, temp_output_dir):
        """Test getting cache info when cache is enabled."""
        cache_dir = temp_output_dir / "cache"
        processor = PipelineProcessor(test_config, cache_dir=cache_dir)

        # Create some fake cache files
        (cache_dir / "file1.pkl").write_bytes(b"x" * 1000)
        (cache_dir / "file2.pkl").write_bytes(b"x" * 2000)

        info = processor.get_cache_info()

        assert info["enabled"] is True
        assert info["cache_dir"] == str(cache_dir)
        assert info["files"] == 2
        assert info["size_mb"] == pytest.approx(3000 / (1024 * 1024), rel=1e-3)

    def test_clear_cache_no_cache(self, test_config):
        """Test clearing cache when cache is disabled."""
        processor = PipelineProcessor(test_config, cache_dir=None)
        cleared = processor.clear_cache()
        assert cleared == 0

    def test_clear_cache_with_cache(self, test_config, temp_output_dir):
        """Test clearing cache when cache is enabled."""
        cache_dir = temp_output_dir / "cache"
        processor = PipelineProcessor(test_config, cache_dir=cache_dir)

        # Create some fake cache files
        cache_files = []
        for i in range(3):
            cache_file = cache_dir / f"file{i}.pkl"
            cache_file.write_bytes(b"test")
            cache_files.append(cache_file)

        cleared = processor.clear_cache()

        assert cleared == 3
        for cache_file in cache_files:
            assert not cache_file.exists()

    def test_get_diagnostics(self, test_config, temp_output_dir):
        """Test getting system diagnostics."""
        processor = PipelineProcessor(test_config, cache_dir=temp_output_dir / "cache")
        diagnostics = processor.get_diagnostics()

        assert "python_version" in diagnostics
        assert "platform" in diagnostics
        assert "cache_enabled" in diagnostics
        assert "cache_dir" in diagnostics
        assert "cache_files" in diagnostics
        assert "cache_size_mb" in diagnostics


# ============================================================================
# Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @patch("larch_cli_wrapper.feff_utils.generate_multi_site_feff_inputs")
    @patch("larch_cli_wrapper.feff_utils.normalize_absorbers")
    @patch("larch_cli_wrapper.feff_utils.run_multi_site_feff_calculations")
    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    @patch("larch.xafs.xftf")
    def test_full_pipeline_single_structure(
        self,
        mock_xftf,
        mock_read_output,
        mock_run_feff,
        mock_normalize,
        mock_generate,
        test_config,
        simple_atoms,
        temp_output_dir,
    ):
        """Test complete pipeline for single structure."""
        # Setup mocks
        mock_normalize.return_value = [0]

        # Create the input file
        input_file = temp_output_dir / "site_0000" / "feff.inp"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text("CONTROL 1 1 1 1 1 1\nPRINT 1 0 0 0 0 3\n")

        mock_input_files = [input_file]
        mock_generate.return_value = mock_input_files
        mock_run_feff.return_value = [(temp_output_dir / "site_0000", True)]

        # Create realistic EXAFS data
        k = np.linspace(0, 15, 100)
        chi = np.sin(k) * np.exp(-k / 10)
        mock_read_output.return_value = (chi, k)

        # Run pipeline
        processor = PipelineProcessor(test_config, cache_dir=None)
        final_group, frame_averages, site_averages, individual_groups = (
            processor.process_trajectory(
                structures=[simple_atoms],  # Single structure as list
                absorber="Au",
                output_dir=temp_output_dir,
                parallel=False,
            )
        )

        # Verify all components were called
        mock_normalize.assert_called_once()
        mock_generate.assert_called_once()
        mock_run_feff.assert_called_once()
        assert mock_read_output.call_count >= 1  # Called by cache and result loading
        mock_xftf.assert_called_once()

        # Verify results
        assert final_group is not None
        assert len(individual_groups) == 1
        assert hasattr(final_group, "k")
        assert hasattr(final_group, "chi")

    @patch("larch_cli_wrapper.feff_utils.generate_multi_site_feff_inputs")
    @patch("larch_cli_wrapper.feff_utils.normalize_absorbers")
    @patch("larch_cli_wrapper.feff_utils.run_multi_site_feff_calculations")
    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    @patch("larch.xafs.xftf")
    @patch("larch_cli_wrapper.exafs_data.create_averaged_group")
    def test_full_pipeline_trajectory(
        self,
        mock_create_avg,
        mock_xftf,
        mock_read_output,
        mock_run_feff,
        mock_normalize,
        mock_generate,
        test_config,
        trajectory_atoms,
        temp_output_dir,
    ):
        """Test complete pipeline for trajectory."""
        # Setup mocks
        mock_normalize.return_value = [0]

        # Mock input generation for each frame - create actual files
        mock_input_files = []
        mock_feff_results = []
        for i in range(len(trajectory_atoms)):
            input_file = temp_output_dir / f"frame_{i:04d}" / "site_0000" / "feff.inp"
            feff_dir = input_file.parent

            # Create the actual file
            input_file.parent.mkdir(parents=True, exist_ok=True)
            input_file.write_text("CONTROL 1 1 1 1 1 1\nPRINT 1 0 0 0 0 3\n")

            mock_input_files.append(input_file)
            mock_feff_results.append((feff_dir, True))

        mock_generate.side_effect = [
            [mock_input_files[i]] for i in range(len(trajectory_atoms))
        ]
        mock_run_feff.return_value = mock_feff_results

        # Create realistic EXAFS data
        k = np.linspace(0, 15, 100)
        chi = np.sin(k) * np.exp(-k / 10)
        mock_read_output.return_value = (chi, k)

        # Mock averaging
        mock_avg_group = Group()
        mock_avg_group.k = k
        mock_avg_group.chi = chi
        mock_create_avg.return_value = mock_avg_group

        # Run pipeline
        processor = PipelineProcessor(test_config, cache_dir=None)
        overall_avg, frame_avg, site_avg, individual = processor.process_trajectory(
            structures=trajectory_atoms,
            absorber="Au",
            output_dir=temp_output_dir,
            parallel=False,
        )

        # Verify results
        assert overall_avg is not None
        assert len(frame_avg) == len(trajectory_atoms)
        assert len(site_avg) == 1  # One site across all frames
        assert len(individual) == len(trajectory_atoms)

        # Verify averaging was called
        assert mock_create_avg.call_count >= 1

    def test_pipeline_error_handling(self, test_config, simple_atoms, temp_output_dir):
        """Test pipeline error handling."""
        processor = PipelineProcessor(test_config, cache_dir=None)

        # Mock failure in FEFF execution
        with patch.object(processor.feff_executor, "execute_batch") as mock_exec:
            mock_exec.return_value = {"task_1": False}  # Task failed

            with patch.object(
                processor.input_generator, "generate_single_site_inputs"
            ) as mock_gen:
                mock_batch = MagicMock()
                mock_batch.tasks = [MagicMock()]
                mock_batch.tasks[0].task_id = "task_1"
                mock_gen.return_value = mock_batch

                final_group, frame_averages, site_averages, individual_groups = (
                    processor.process_trajectory(
                        structures=[simple_atoms],  # Single structure as list
                        absorber="Au",
                        output_dir=temp_output_dir,
                    )
                )

                # Should handle failed FEFF execution gracefully
                assert final_group is None
                assert len(individual_groups) == 0
