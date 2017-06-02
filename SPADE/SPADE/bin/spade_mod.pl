#!/usr/bin/perl -w

#$CHP = "your-path-to-Charniak's-parser-directory";
$CHP = "/Users/hemantnigam/Documents/workspace/nlp/project/rnn-nlu/SPADE/SPADE/parser/bllip-parser/first-stage/PARSE";
#$CHP = "/nfs/isd/radu/Work/Parsing/CharniakParser/";

if( scalar(@ARGV)!= 1 && scalar(@ARGV)!= 2 ){
    print STDERR "Usage: spade.pl [-seg-only] one-sent-per-line-file\n";
}
else{
    if( $CHP eq "your-path-to-Charniak's-parser-directory" ){
	print STDERR "You need to set the path to Charniak's parser directory first.\n" and exit;
    }

    $argv = shift;
    if( $argv eq "-seg-only" ){
	$argv = shift;
	@args = ("$CHP/parseIt $CHP/DATA/ $argv > $argv.chp");
	#print STDERR "Charniak's syntactic parser in progress...\n";
	system(@args) == 0
	    or die "system @args failed: $?";
    #print STDERR "Done.\n";

	@args = ("perl", "edubreak.pl", "$argv.chp");
	system(@args) == 0
	    or die "system @args failed: $?";
    }
    else{
	@args = ("$CHP/parseIt $CHP/DATA/ $argv > $argv.chp");
	#print STDERR "Charniak's syntactic parser in progress...\n";
	system(@args) == 0
	    or die "system @args failed: $?";
    #print STDERR "Done.\n";

	@args = ("perl", "discparse.pl", "$argv.chp");
	system(@args) == 0
	    or die "system @args failed: $?";

	`rm -f $argv.chp.edu.fmt`;
	`rm -f $argv.chp.edu`;
    }
    `rm -f $argv.chp`;

}
