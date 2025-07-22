#!/usr/bin/env perl
use strict;
use warnings;

# Usage: ./patch_casts_and_wrap.pl file.cpp

my $filename = shift 
  or die "Usage: $0 <file.cpp>\n";

# Read the entire file
open my $in, '<', $filename 
  or die "Can't read $filename: $!";
my @lines = <$in>;
close $in;

# The macro to insert once
my $inserted_macro = 0;
my $hip_check_macro = <<'MACRO';
#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = call;                                                   \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at "       \
                      << __FILE__ << ":" << __LINE__ << " in " << #call << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

MACRO

# Patterns
my $cast_funcs = qr{hipHost(?:Alloc|Malloc|GetDevicePointer)};
my $wrap_funcs = qr{hip(?:SetDeviceFlags|Host(?:Alloc|Malloc|Free|GetDevicePointer)|
                    Malloc|Free|Memcpy(?:Async)?|Memset(?:Async)?|
                    Event(?:Create|Record|Destroy|ElapsedTime)|Stream(?:Create|Synchronize))}x;

my @output;

foreach my $line (@lines) {
    # 1) Insert HIP_CHECK macro once, after the last #include
    if (!$inserted_macro && $line =~ /^\s*#include/) {
        push @output, $line;
        next;
    }
    if (!$inserted_macro && $line !~ /^\s*#/) {
        push @output, "\n$hip_check_macro";
        $inserted_macro = 1;
        # fall through to process this line as code
    }

    # 2) Cast patch: hipHostAlloc, hipHostMalloc, hipHostGetDevicePointer
    unless ($line =~ /\b$cast_funcs\s*\(\s*\(void\s*\*\*\)\s*&/) {
        $line =~ s{
            \b($cast_funcs)           # Function name
            \s* \( \s* & ([A-Za-z_]\w*)  # &var
        }{
            "$1((void **)&$2"         # Add cast
        }xeg;
    }

    # 3) Wrap single-call lines with HIP_CHECK(...)
    if ($line =~ /\b$wrap_funcs\s*\(/ && $line !~ /HIP_CHECK/) {
        # only wrap if the entire line is exactly one HIP call terminating in ';'
        if ($line =~ /^\s*($wrap_funcs\s*\(.*\));\s*$/) {
            my $call = $1;
            $line = "HIP_CHECK($call);\n";
        }
    }

    push @output, $line;
}

# 4) Write back
open my $out, '>', $filename 
  or die "Can't write $filename: $!";
print $out @output;
close $out;