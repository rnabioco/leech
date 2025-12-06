"""Tests for BED file reading and region-based position finding."""

from pathlib import Path

import pytest

from leech.io.bed_reader import BedIndex, BedRegion, BedRegionSearcher, load_bed_regions


class TestBedRegion:
    """Test BedRegion data class."""

    def test_center_calculation_even(self):
        """Test center position calculation for even-length region."""
        region = BedRegion(chrom="chr1", start=100, end=110, name="test")
        assert region.center == 105

    def test_center_calculation_odd(self):
        """Test center position calculation for odd-length region."""
        region = BedRegion(chrom="chr1", start=100, end=111, name="test")
        assert region.center == 105  # (100 + 111) // 2

    def test_center_single_base(self):
        """Test center for single-base region."""
        region = BedRegion(chrom="chr1", start=100, end=101, name="test")
        assert region.center == 100

    def test_default_name(self):
        """Test that name defaults to None."""
        region = BedRegion(chrom="chr1", start=100, end=110)
        assert region.name is None


class TestBedIndex:
    """Test BedIndex class."""

    def test_empty_index(self):
        """Test empty index."""
        index = BedIndex()
        assert index.total_regions() == 0
        assert index.get_all_chromosomes() == []

    def test_add_region(self):
        """Test adding regions."""
        index = BedIndex()
        region = BedRegion(chrom="chr1", start=100, end=110, name="test")
        index.add_region(region)

        assert index.total_regions() == 1
        assert "chr1" in index.get_all_chromosomes()

    def test_add_multiple_regions(self):
        """Test adding multiple regions to same chromosome."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="r1"))
        index.add_region(BedRegion(chrom="chr1", start=200, end=210, name="r2"))
        index.add_region(BedRegion(chrom="chr2", start=50, end=60, name="r3"))

        assert index.total_regions() == 3
        assert len(index.regions["chr1"]) == 2
        assert len(index.regions["chr2"]) == 1

    def test_get_overlapping_regions_single(self):
        """Test finding a single overlapping region."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="target"))
        index.add_region(BedRegion(chrom="chr1", start=200, end=210, name="other"))

        # Query that overlaps target
        result = index.get_overlapping_regions("chr1", 95, 115)
        assert len(result) == 1
        assert result[0].name == "target"

    def test_get_overlapping_regions_multiple(self):
        """Test finding multiple overlapping regions."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=120, name="r1"))
        index.add_region(BedRegion(chrom="chr1", start=110, end=130, name="r2"))
        index.add_region(BedRegion(chrom="chr1", start=200, end=210, name="r3"))

        # Query that overlaps r1 and r2
        result = index.get_overlapping_regions("chr1", 105, 125)
        assert len(result) == 2
        names = {r.name for r in result}
        assert names == {"r1", "r2"}

    def test_get_overlapping_regions_no_match(self):
        """Test when no regions overlap."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="r1"))

        result = index.get_overlapping_regions("chr1", 200, 300)
        assert len(result) == 0

    def test_get_overlapping_regions_wrong_chrom(self):
        """Test when querying wrong chromosome."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="r1"))

        result = index.get_overlapping_regions("chr2", 100, 110)
        assert len(result) == 0

    def test_get_overlapping_edge_cases(self):
        """Test edge cases for overlap detection."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="target"))

        # Query ends exactly at region start (no overlap)
        result = index.get_overlapping_regions("chr1", 90, 100)
        assert len(result) == 0

        # Query starts exactly at region end (no overlap)
        result = index.get_overlapping_regions("chr1", 110, 120)
        assert len(result) == 0

        # Query touches region by 1 base
        result = index.get_overlapping_regions("chr1", 99, 101)
        assert len(result) == 1


class TestLoadBedRegions:
    """Test BED file loading."""

    @pytest.fixture
    def sample_bed_file(self, tmp_path):
        """Create a sample BED file."""
        bed_content = """chr1\t100\t110\tregion1
chr1\t200\t220\tregion2
chr1\t500\t510\t.
chr2\t100\t150\tregion3
"""
        bed_file = tmp_path / "test.bed"
        bed_file.write_text(bed_content)
        return bed_file

    @pytest.fixture
    def bed3_file(self, tmp_path):
        """Create a BED3 file (no name column)."""
        bed_content = """chr1\t100\t110
chr1\t200\t220
"""
        bed_file = tmp_path / "test3.bed"
        bed_file.write_text(bed_content)
        return bed_file

    def test_load_bed_file(self, sample_bed_file):
        """Test loading BED file."""
        bed_index = load_bed_regions(sample_bed_file)

        assert "chr1" in bed_index.regions
        assert "chr2" in bed_index.regions
        assert len(bed_index.regions["chr1"]) == 3
        assert bed_index.total_regions() == 4

    def test_load_bed3_file(self, bed3_file):
        """Test loading BED3 file without name column."""
        bed_index = load_bed_regions(bed3_file)

        assert bed_index.total_regions() == 2
        # All regions should have None as name
        for region in bed_index.regions["chr1"]:
            assert region.name is None

    def test_load_bed_dot_name(self, sample_bed_file):
        """Test that '.' in name field is converted to None."""
        bed_index = load_bed_regions(sample_bed_file)

        # Find the region at 500-510 which has '.' as name
        for region in bed_index.regions["chr1"]:
            if region.start == 500:
                assert region.name is None

    def test_load_bed_with_comments(self, tmp_path):
        """Test loading BED file with comments and headers."""
        bed_content = """# This is a comment
track name="test"
browser position chr1:1-1000
chr1\t100\t110\tregion1
"""
        bed_file = tmp_path / "test.bed"
        bed_file.write_text(bed_content)

        bed_index = load_bed_regions(bed_file)
        assert bed_index.total_regions() == 1

    def test_load_empty_bed(self, tmp_path):
        """Test loading empty BED file."""
        bed_file = tmp_path / "empty.bed"
        bed_file.write_text("")

        bed_index = load_bed_regions(bed_file)
        assert bed_index.total_regions() == 0

    def test_load_nonexistent_file(self, tmp_path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_bed_regions(tmp_path / "nonexistent.bed")

    def test_load_invalid_format_too_few_columns(self, tmp_path):
        """Test error with invalid format (too few columns)."""
        bed_file = tmp_path / "invalid.bed"
        bed_file.write_text("chr1\t100\n")

        with pytest.raises(ValueError, match="expected at least 3 fields"):
            load_bed_regions(bed_file)

    def test_load_invalid_format_bad_coordinates(self, tmp_path):
        """Test error with invalid coordinates."""
        bed_file = tmp_path / "invalid.bed"
        bed_file.write_text("chr1\tabc\t110\tregion1\n")

        with pytest.raises(ValueError, match="could not parse coordinates"):
            load_bed_regions(bed_file)

    def test_load_invalid_format_negative_start(self, tmp_path):
        """Test error with negative start position."""
        bed_file = tmp_path / "invalid.bed"
        bed_file.write_text("chr1\t-10\t110\tregion1\n")

        with pytest.raises(ValueError, match="cannot be negative"):
            load_bed_regions(bed_file)

    def test_load_invalid_format_end_before_start(self, tmp_path):
        """Test error when end <= start."""
        bed_file = tmp_path / "invalid.bed"
        bed_file.write_text("chr1\t100\t100\tregion1\n")

        with pytest.raises(ValueError, match="end must be greater than start"):
            load_bed_regions(bed_file)


class TestBedRegionSearcher:
    """Test BED-based position finding."""

    @pytest.fixture
    def bed_index(self):
        """Create a BedIndex for testing."""
        index = BedIndex()
        index.add_region(BedRegion(chrom="chr1", start=100, end=110, name="region1"))
        index.add_region(BedRegion(chrom="chr1", start=200, end=220, name="region2"))
        index.add_region(BedRegion(chrom="chr1", start=500, end=510, name=None))  # '.' name
        index.add_region(BedRegion(chrom="chr2", start=100, end=150, name="region3"))
        return index

    @pytest.fixture
    def mock_alignment(self):
        """Create a mock alignment object."""

        class MockAlignment:
            def __init__(self, ref_name, ref_start, ref_end, cigartuples):
                self.reference_name = ref_name
                self.reference_start = ref_start
                self.reference_end = ref_end
                self.cigartuples = cigartuples

        # Simple alignment covering chr1:50-250 with perfect match
        return MockAlignment(
            ref_name="chr1",
            ref_start=50,
            ref_end=250,
            cigartuples=[(0, 200)],  # 200M
        )

    def test_find_positions_basic(self, bed_index, mock_alignment):
        """Test finding positions from BED regions."""
        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="A" * 200,
            alignment=mock_alignment,
        )

        # Should find region1 (center=105) and region2 (center=210)
        # region at 500-510 is outside alignment
        assert len(results) == 2

        positions = [pos for pos, label in results]
        # Query positions: center - ref_start = 105-50=55, 210-50=160
        assert 55 in positions
        assert 160 in positions

    def test_find_positions_with_labels(self, bed_index, mock_alignment):
        """Test that labels come from BED name field."""
        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="A" * 200,
            alignment=mock_alignment,
        )

        labels_by_pos = {pos: label for pos, label in results}
        assert labels_by_pos[55] == "region1"
        assert labels_by_pos[160] == "region2"

    def test_default_label_fallback(self, bed_index):
        """Test fallback to default label for regions with None name."""

        class MockAlignment:
            reference_name = "chr1"
            reference_start = 450
            reference_end = 600
            cigartuples = [(0, 150)]  # 150M

        searcher = BedRegionSearcher(bed_index, default_label="my_default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="A" * 150,
            alignment=MockAlignment(),
        )

        # Should find region at 500-510 with center=505
        assert len(results) == 1
        pos, label = results[0]
        assert label == "my_default"

    def test_no_alignment(self, bed_index):
        """Test when alignment is None."""
        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="ACGT" * 50,
            alignment=None,
        )

        assert len(results) == 0

    def test_no_overlapping_regions(self, bed_index):
        """Test when read doesn't overlap any BED regions."""

        class MockAlignment:
            reference_name = "chr1"
            reference_start = 300
            reference_end = 400
            cigartuples = [(0, 100)]

        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="A" * 100,
            alignment=MockAlignment(),
        )

        assert len(results) == 0

    def test_wrong_chromosome(self, bed_index, mock_alignment):
        """Test when read is on different chromosome than BED regions."""

        class MockAlignment:
            reference_name = "chr3"
            reference_start = 100
            reference_end = 200
            cigartuples = [(0, 100)]

        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="A" * 100,
            alignment=MockAlignment(),
        )

        assert len(results) == 0

    def test_center_outside_alignment(self, bed_index):
        """Test when BED region overlaps but center is outside alignment."""

        class MockAlignment:
            reference_name = "chr1"
            # Alignment covers 100-103, region1 is 100-110 with center 105
            reference_start = 100
            reference_end = 103
            cigartuples = [(0, 3)]

        searcher = BedRegionSearcher(bed_index, default_label="default")

        results = searcher.find_positions_with_labels(
            read_id="test_read",
            sequence="ACG",
            alignment=MockAlignment(),
        )

        # Center (105) is outside alignment (100-103), so should find nothing
        assert len(results) == 0


class TestMutualExclusion:
    """Test that --bed and --motif are mutually exclusive."""

    def test_both_specified_raises_error(self, tmp_path):
        """Test error when both BED and motif specified in orchestrator."""
        from leech.preparation.orchestrator import prepare_training_data_with_split

        bed_file = tmp_path / "test.bed"
        bed_file.write_text("chr1\t100\t110\ttest\n")

        with pytest.raises(ValueError, match="Cannot specify both"):
            prepare_training_data_with_split(
                pod5_path=Path("nonexistent.pod5"),
                bam_path=Path("nonexistent.bam"),
                output_dir=tmp_path / "output",
                motif="CCAGGC",
                bed_path=bed_file,
            )

    def test_both_specified_raises_error_parallel(self, tmp_path):
        """Test error when both BED and motif specified in parallel."""
        from leech.preparation.parallel import prepare_training_data_parallel

        bed_file = tmp_path / "test.bed"
        bed_file.write_text("chr1\t100\t110\ttest\n")

        with pytest.raises(ValueError, match="Cannot specify both"):
            prepare_training_data_parallel(
                bam_path=Path("nonexistent.bam"),
                pod5_path=Path("nonexistent.pod5"),
                motif="CCAGGC",
                bed_path=bed_file,
            )

    def test_both_specified_raises_error_handler(self, tmp_path):
        """Test error when both BED and motif specified in command handler."""
        from leech.commands.prepare import handle_prepare

        bed_file = tmp_path / "test.bed"
        bed_file.write_text("chr1\t100\t110\ttest\n")

        with pytest.raises(ValueError, match="Cannot specify both"):
            handle_prepare(
                pod5=Path("nonexistent.pod5"),
                bam=Path("nonexistent.bam"),
                output_dir=tmp_path / "output",
                motif="CCAGGC",
                bed=bed_file,
            )
