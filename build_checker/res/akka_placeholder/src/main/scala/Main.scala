import akka.actor.{Actor, ActorSystem, Props}

// Message to increment the counter
case object Increment

// Actor that keeps track of how many increments it has seen
class CountingActor extends Actor {
  // Internal counter, starts at zero
  private var count = 0

  def receive: Receive = {
    // If we get Increment, increase count and log it
    case Increment =>
      count += 1
      println(s"CountingActor: Count is now $count")
    // If unknown message, log that it's unknown
    case _ =>
      println("CountingActor: Received unknown message.")
  }
}

object CountingActorApp extends App {
  // Create an ActorSystem named "CountingSystem"
  val system = ActorSystem("CountingSystem")
  // Create an instance of CountingActor
  val counter = system.actorOf(Props[CountingActor](), "counter")

  // Send multiple increments
  counter ! Increment
  counter ! Increment
  counter ! Increment

  Thread.sleep(500)
  system.terminate()
}