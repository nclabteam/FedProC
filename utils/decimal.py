class Decimal:
    # TODO: # OPTIMISE: code to maximize performance
    """
    Class decimal, a more reliable alternative to float. | v0.1
    ============================================================
            Python's floats (and in many other languages as well) are
    pretty inaccurate. While on the outside it may look like this:

    .1 + .1 + .1

            But on the inside, it gets converted to base 2. It tells
    the computer, "2 to the power of what is 0.1?". The
    computer says, "Oh, I don't know; would an approximation
    be sufficient?"
    Python be like, "Oh, sure, why not? It's not like we need to
    give it that much accuracy."
            And so that happens. But what they ARE good at is
    everything else, including multiplying a float and a
    10 together. So I abused that and made this: the decimal
    class. Us humans knows that 1 + 1 + 1 = 3. Well, most of us
    anyway but that's not important. The thing is, computers can
    too! This new replacement does the following:

            1. Find how many 10 ^ n it takes to get the number inputted
                    into a valid integer.
            2. Make a list with the original float and n (multiplying the by
                    10^-n is inaccurate)

            And that's pretty much it, if you don't count the
    adding, subtracting, etc algorithm. This is more accurate than just
    ".1 + .1 + .1". But then, it's more simple than hand-typing
    (.1 * 100 + .01 * 100 + .1 * 100)/100
    (which is basically the algorithm for this). But it does have it's costs.
    --------------------------------------------------------------------------

    BAD #1: It's slightly slower then the conventional .1 + .1 + .1 but
        it DOES make up for accuracy

    BAD #2: It's useless, there are many libraries out there that solves the
            same problem as this. They may be more or less efficient than this
            method. Thus rendering this useless.
    --------------------------------------------------------------------------
    And that's pretty much it! Thanks for stopping by to read this doc-string.
    --------------------------------------------------------------------------
        Copyright © 2020 Bryan Hu

        Permission is hereby granted, free of charge, to any person obtaining
        a copy of this software and associated documentation files
        (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify,
        merge, publish, distribute, sub-license, and/or sell copies of
        the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included
        in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
        OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def __init__(self, number):
        super(Decimal, self).__init__()
        if number is iter:
            processed = float(number[0])
        else:
            processed = float(number)
        x = 10
        while round(processed * x) != processed * x:
            x *= 10
        self.number = [processed, x]

    def __add__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal((num[0] * maximum + the_other_number[0] * maximum) / maximum)
        except IndexError:
            raise "Entered {}, which has the type {},\
             is not a valid type".format(
                other, type(other)
            )

    def __float__(self):
        return float(self.number[0])

    def __bool__(self):
        return bool(self.number[0])

    def __str__(self):
        return str(self.number)

    def __iter__(self):
        return (x for x in self.number)

    def __repr__(self):
        return str(self.number[0])

    def __sub__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal((num[0] * maximum - the_other_number[0] * maximum) / maximum)
        except IndexError:
            raise "Entered {}, which has the type {},\
         is not a valid type".format(
                other, type(other)
            )

    def __div__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal(
                ((num[0] * maximum) / (the_other_number[0] * maximum)) / maximum
            )
        except IndexError:
            raise "Entered {}, which has the type {},\
         is not a valid type".format(
                other, type(other)
            )

    def __floordiv__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal(
                ((num[0] * maximum) // (the_other_number[0] * maximum)) / maximum
            )
        except IndexError:
            raise "Entered {}, which has the type {},\
         is not a valid type".format(
                other, type(other)
            )

    def __mul__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal(
                ((num[0] * maximum) * (the_other_number[0] * maximum)) / maximum
            )
        except IndexError:
            raise "Entered {}, which has the type {},\
         is not a valid type".format(
                other, type(other)
            )

    def __mod__(self, other):
        the_other_number, num = list(other), list(self.number)
        try:
            maximum = max(float(num[1]), float(the_other_number[1]))
            return Decimal(
                ((num[0] * maximum) % (the_other_number[0] * maximum)) / maximum
            )
        except IndexError:
            raise "Entered {}, which has the type {},\
         is not a valid type".format(
                other, type(other)
            )
